"""The TerraMind agent's three skills, independent of the A2A transport.

* ``embed_scene``        fetch 6 Sentinel-2 bands, run TerraMind, keep the full tensor
                         server-side under an ``embedding_id`` and return only metadata + stats.
* ``compare_embeddings`` cosine similarity of two stored embeddings, overall and per tile.
* ``rank_similar``       rank stored embeddings by similarity to a reference.

The full (196, 192) tensor never goes into an LLM's context: callers pass ``embedding_id``s
around. The store is a dict backed by files (``data/embeddings/<id>.npz`` + ``<id>.json``), so
embeddings survive an agent restart and every one records the session that created it; a vector
store is the natural next step. ``torch`` / ``terratorch`` / ``rasterio`` are imported lazily so the
comparison logic can be tested without them.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
MODEL_NAME = "ibm-esa-geospatial/TerraMind-1.0-tiny (terratorch_terramind_v1_tiny)"
DEFAULT_PATCH = 224
MAX_NODATA = 0.2  # refuse a crop that is more than this fraction no-data (all six bands == 0)
TOKEN_PX = 16  # ViT patch size: a 224 px crop gives a 14 x 14 grid of tiles (196 tokens)

# TerraMind's documented 6-band Sentinel-2 L2A subset, in order, and the Earth Search asset keys.
BAND_NAMES = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
ASSET_KEYS = dict(zip(BAND_NAMES, ["blue", "green", "red", "nir08", "swir16", "swir22"]))


class SkillError(ValueError):
    """A request the agent can't satisfy (unknown id, bad argument, scene not found)."""


class EmbeddingStore:
    """id -> {"tokens": (n_tokens, dim) float32, "meta": {...}}, thread-safe.

    With ``directory`` set, every put is also written to ``<id>.npz`` (tokens) and ``<id>.json``
    (metadata), and ``get`` falls back to disk, so embeddings outlive the process.
    """

    def __init__(self, directory: Optional[str | Path] = None) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}
        self._dir = Path(directory) if directory else None
        self._lock = threading.Lock()

    def put(self, embedding_id: str, tokens: np.ndarray, meta: Dict[str, Any]) -> None:
        tokens = tokens.astype(np.float32)
        with self._lock:
            self._items[embedding_id] = {"tokens": tokens, "meta": meta}
            if self._dir:
                self._dir.mkdir(parents=True, exist_ok=True)
                tmp = self._dir / f"{embedding_id}.tmp.npz"
                np.savez_compressed(tmp, tokens=tokens)
                tmp.replace(self._dir / f"{embedding_id}.npz")
                (self._dir / f"{embedding_id}.json").write_text(json.dumps(meta, indent=1))

    def _load(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        if not self._dir:
            return None
        npz, js = self._dir / f"{embedding_id}.npz", self._dir / f"{embedding_id}.json"
        if not (npz.exists() and js.exists()):
            return None
        return {"tokens": np.load(npz)["tokens"], "meta": json.loads(js.read_text())}

    def get(self, embedding_id: str) -> Dict[str, Any]:
        with self._lock:
            item = self._items.get(embedding_id)
            if item is None and "/" not in embedding_id and ".." not in embedding_id:
                item = self._load(embedding_id)
                if item is not None:
                    self._items[embedding_id] = item
        if item is None:
            known = self.ids()
            raise SkillError(f"unknown embedding_id {embedding_id!r}; known ids: {known or 'none (call embed_scene first)'}")
        return item

    def ids(self) -> List[str]:
        with self._lock:
            names = set(self._items)
            if self._dir and self._dir.exists():
                names |= {p.stem for p in self._dir.glob("*.json")}
        return sorted(names)


STORE = EmbeddingStore(os.environ.get("TERRAMIND_STORE_DIR", str(Path(__file__).resolve().parent.parent / "data" / "embeddings")))

# ── model + data (heavy, lazy) ───────────────────────────────────────────────

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from terratorch import BACKBONE_REGISTRY

            _model = BACKBONE_REGISTRY.build(
                "terratorch_terramind_v1_tiny", pretrained=True,
                modalities=["S2L2A"], bands={"S2L2A": BAND_NAMES},
            )
            _model.eval()
        return _model


def _fetch_scene(collection: str, item_id: str, patch_size: int):
    """(patch (6, p, p) reflectance, STAC item, crop bbox [w, s, e, n]). All bands are read over the *same* geographic
    window (the centred crop of the 10 m blue band), so the 20 m SWIR bands stay co-registered."""
    import rasterio
    from pystac_client import Client
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    item = Client.open(EARTH_SEARCH_URL).get_collection(collection).get_item(item_id)
    if item is None:
        raise SkillError(f"STAC item {item_id!r} not found in collection {collection!r}")
    missing = [ASSET_KEYS[b] for b in BAND_NAMES if ASSET_KEYS[b] not in item.assets]
    if missing:
        raise SkillError(f"item {item_id!r} lacks assets {missing} (needed for TerraMind S2L2A)")

    with rasterio.open(item.assets[ASSET_KEYS["BLUE"]].href) as ref:
        cx, cy, half = ref.width // 2, ref.height // 2, patch_size // 2
        win = Window(cx - half, cy - half, patch_size, patch_size)
        bounds = rasterio.windows.bounds(win, ref.transform)
        crs = ref.crs
    arrays = []
    for band in BAND_NAMES:
        with rasterio.open(item.assets[ASSET_KEYS[band]].href) as src:
            data = src.read(1, window=from_bounds(*bounds, transform=src.transform),
                            out_shape=(patch_size, patch_size), resampling=Resampling.bilinear)
        arrays.append(np.asarray(data, dtype=np.float32) / 10000.0)  # L2A reflectance scale
    crop_bbox = [round(v, 5) for v in transform_bounds(crs, "EPSG:4326", *bounds)]
    return np.stack(arrays), item, crop_bbox


def _run_model(patch: np.ndarray) -> np.ndarray:
    """(6, p, p) -> (n_tokens, dim)."""
    import torch

    with torch.no_grad():
        out = _get_model()({"S2L2A": torch.from_numpy(patch).unsqueeze(0)})
    if isinstance(out, (list, tuple)):
        out = out[-1]
    return out.float().squeeze(0).cpu().numpy()


# ── skills ───────────────────────────────────────────────────────────────────


def _tile_id(item) -> Optional[str]:
    p = item.properties
    code = p.get("grid:code") or p.get("s2:mgrs_tile") or p.get("mgrs:latitude_band")
    if code:
        return str(code).replace("MGRS-", "")
    parts = item.id.split("_")  # S2B_43QHV_20240113_0_L2A
    return parts[1] if len(parts) > 2 else None


def make_embedding_id(item_id: str, patch_size: int) -> str:
    return f"emb_{item_id}" + ("" if patch_size == DEFAULT_PATCH else f"_p{patch_size}")


def embed_scene(
    collection: str,
    item_id: str,
    patch_size: int = DEFAULT_PATCH,
    *,
    session_id: Optional[str] = None,
    fetch: Callable = _fetch_scene,
    run: Callable = _run_model,
) -> Dict[str, Any]:
    if patch_size % TOKEN_PX or not 32 <= patch_size <= 512:
        raise SkillError(f"patch_size must be a multiple of {TOKEN_PX} between 32 and 512, got {patch_size}")
    patch, item, crop_bbox = fetch(collection, item_id, patch_size)
    nodata = float(np.mean(~patch.any(axis=0)))
    if nodata > MAX_NODATA:
        # Scenes at the swath edge have no pixels at the tile centre; embedding zeros would give
        # meaningless (and mutually identical) vectors, so say so instead.
        raise SkillError(
            f"{item_id}: {nodata:.0%} of the centre crop is no-data (the scene covers this tile only "
            "partially). Pick another scene over the area."
        )
    tokens = run(patch)
    embedding_id = make_embedding_id(item_id, patch_size)
    props = item.properties
    meta = {
        "embedding_id": embedding_id,
        "collection": collection,
        "item_id": item_id,
        "datetime": props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "tile_id": _tile_id(item),
        "crop_bbox": crop_bbox,
        "patch_size": patch_size,
        "nodata_fraction": round(nodata, 4),
        "session_id": session_id,  # the chat session that created it (traceability)
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid": [int(round(math.sqrt(tokens.shape[0])))] * 2,
    }
    STORE.put(embedding_id, tokens, meta)
    return {
        **meta,
        "bands_used": BAND_NAMES,
        "model": MODEL_NAME,
        "embedding_shape": list(tokens.shape),
        "embedding_mean": float(tokens.mean()),
        "embedding_std": float(tokens.std()),
        "embedding_l2_norm": float(np.linalg.norm(tokens)),
        "note": "The full tensor is stored server-side; pass embedding_id to compare_embeddings / rank_similar.",
    }


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity along the last axis."""
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(na * nb, 1e-12)


def _compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    ta, tb = a["tokens"], b["tokens"]
    ma, mb = a["meta"], b["meta"]
    out: Dict[str, Any] = {
        "cosine_similarity": round(float(_cos(ta.mean(0), tb.mean(0))), 4),
        "scene_a": {k: ma.get(k) for k in ("embedding_id", "item_id", "datetime", "cloud_cover", "tile_id")},
        "scene_b": {k: mb.get(k) for k in ("embedding_id", "item_id", "datetime", "cloud_cover", "tile_id")},
    }
    same_tile = bool(ma.get("tile_id")) and ma.get("tile_id") == mb.get("tile_id")
    same_grid = ta.shape == tb.shape
    out["same_footprint"] = same_tile and same_grid
    if not out["same_footprint"]:
        out["per_tile"] = None
        out["note"] = (
            "Scenes do not share a tile id (or crop size), so crops cover different ground; "
            "only the overall similarity is meaningful, not the per-tile comparison."
        )
        return out
    sims = _cos(ta, tb)  # (n_tokens,)
    g = ma["grid"][1]
    worst = np.argsort(sims)[:3]
    out["per_tile"] = {
        "mean": round(float(sims.mean()), 4),
        "min": round(float(sims.min()), 4),
        "max": round(float(sims.max()), 4),
        "least_similar_tiles": [
            {"row": int(i // g), "col": int(i % g), "similarity": round(float(sims[i]), 4)} for i in worst
        ],
        "grid": ma["grid"],
    }
    out["note"] = "Same tile id: per-tile similarity shows where the two scenes differ (row/col in the grid, 0-based from top-left)."
    return out


def compare_embeddings(id_a: str, id_b: str, *, store: EmbeddingStore = STORE) -> Dict[str, Any]:
    return _compare(store.get(id_a), store.get(id_b))


def rank_similar(reference_id: str, candidate_ids: List[str], *, store: EmbeddingStore = STORE) -> Dict[str, Any]:
    if not candidate_ids:
        raise SkillError("candidate_ids is empty")
    ref = store.get(reference_id)
    rows = []
    for cid in candidate_ids:
        cmp = _compare(ref, store.get(cid))
        rows.append({
            "embedding_id": cid,
            "item_id": cmp["scene_b"]["item_id"],
            "datetime": cmp["scene_b"]["datetime"],
            "cloud_cover": cmp["scene_b"]["cloud_cover"],
            "cosine_similarity": cmp["cosine_similarity"],
            "same_footprint": cmp["same_footprint"],
            "mean_tile_similarity": (cmp["per_tile"] or {}).get("mean"),
        })
    rows.sort(key=lambda r: r["cosine_similarity"], reverse=True)
    return {"reference_id": reference_id, "ranking": rows}


SKILLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "embed_scene": embed_scene,
    "compare_embeddings": compare_embeddings,
    "rank_similar": rank_similar,
}


def _intify(obj: Any) -> Any:
    """A2A data parts are protobuf Structs: every number arrives as a float (224 -> 224.0)."""
    if isinstance(obj, dict):
        return {k: _intify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_intify(v) for v in obj]
    return int(obj) if isinstance(obj, float) and obj.is_integer() else obj


def run_skill(skill: str, args: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    args = _intify(args)
    if skill == "embed_scene":
        args = {**args, "session_id": session_id}  # set by the server, never by the model
    fn = SKILLS.get(skill)
    if fn is None:
        raise SkillError(f"unknown skill {skill!r}; available: {sorted(SKILLS)}")
    try:
        return fn(**args)
    except TypeError as exc:  # wrong/missing argument names
        raise SkillError(f"bad arguments for {skill}: {exc}") from exc
