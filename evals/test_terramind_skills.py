"""Offline tests for the TerraMind agent's skills (no torch, no network).

Run: python -m evals.test_terramind_skills
"""

from types import SimpleNamespace

import numpy as np

from terramind_agent import skills
from terramind_agent.skills import EmbeddingStore, SkillError

RNG = np.random.default_rng(0)


def item(item_id, tile="43QHV", date="2024-01-13T05:24:02Z", cloud=1.5):
    return SimpleNamespace(id=item_id, properties={"datetime": date, "eo:cloud_cover": cloud, "grid:code": f"MGRS-{tile}"})


def fake_fetch(patch_by_id):
    def fetch(collection, item_id, patch_size):
        _, tile, day, *_ = item_id.split("_")
        date = f"{day[:4]}-{day[4:6]}-{day[6:]}T05:24:02Z"
        return patch_by_id[item_id], item(item_id, tile=tile, date=date), [78.0, 17.0, 78.1, 17.1]
    return fetch


def fake_run(patch):  # deterministic "model": 196 tokens x 8 dims derived from the patch
    flat = patch.reshape(6, -1)[:, :196].T  # (196, 6)
    return np.concatenate([flat, flat[:, :2]], axis=1).astype(np.float32)


def embed(item_id, patch, store=None):
    skills.STORE = store or skills.STORE
    return skills.embed_scene("sentinel-2-l2a", item_id, fetch=fake_fetch({item_id: patch}), run=fake_run)


def fresh_store():
    st = EmbeddingStore()
    skills.STORE = st
    return st


def test_embed_stores_tensor_and_returns_metadata_only():
    st = fresh_store()
    out = embed("S2B_43QHV_20240113_0_L2A", RNG.random((6, 224, 224), dtype=np.float32) + 0.1, st)
    assert out["embedding_id"] == "emb_S2B_43QHV_20240113_0_L2A"
    assert out["tile_id"] == "43QHV" and out["cloud_cover"] == 1.5 and out["datetime"].startswith("2024-01-13")
    assert out["embedding_shape"] == [196, 8] and out["grid"] == [14, 14]
    assert st.get(out["embedding_id"])["tokens"].shape == (196, 8)
    assert "tokens" not in out and all(not isinstance(v, np.ndarray) for v in out.values())


def test_nodata_crop_is_rejected():
    fresh_store()
    try:
        embed("S2A_43QHV_20240331_0_L2A", np.zeros((6, 224, 224), dtype=np.float32))
    except SkillError as exc:
        assert "no-data" in str(exc)
    else:
        raise AssertionError("expected SkillError")


def test_bad_patch_size_rejected():
    for ps in (100, 16, 1024):
        try:
            skills.embed_scene("c", "i", ps)
        except SkillError:
            continue
        raise AssertionError(ps)


def test_compare_same_tile_has_per_tile_view():
    st = fresh_store()
    a = RNG.random((6, 224, 224), dtype=np.float32) + 0.1
    b = a.copy()
    b[:, :16, 112:128] = RNG.random((6, 16, 16), dtype=np.float32) + 0.1  # change one grid cell...
    embed("S2B_43QHV_20240113_0_L2A", a, st)
    embed("S2A_43QHV_20240328_0_L2A", b, st)
    out = skills.compare_embeddings("emb_S2B_43QHV_20240113_0_L2A", "emb_S2A_43QHV_20240328_0_L2A", store=st)
    assert out["same_footprint"] is True and out["per_tile"] is not None
    assert 0 < out["cosine_similarity"] <= 1
    assert out["per_tile"]["max"] >= out["per_tile"]["mean"] >= out["per_tile"]["min"]
    assert len(out["per_tile"]["least_similar_tiles"]) == 3
    assert out["scene_b"]["datetime"].startswith("2024-03-28")


def test_compare_different_tiles_says_not_same_footprint():
    st = fresh_store()
    embed("S2B_43QHV_20240113_0_L2A", RNG.random((6, 224, 224), dtype=np.float32) + 0.1, st)
    embed("S2B_44QKE_20240113_0_L2A", RNG.random((6, 224, 224), dtype=np.float32) + 0.1, st)
    out = skills.compare_embeddings("emb_S2B_43QHV_20240113_0_L2A", "emb_S2B_44QKE_20240113_0_L2A", store=st)
    assert out["same_footprint"] is False and out["per_tile"] is None and "tile" in out["note"]


def test_identical_scenes_have_similarity_one():
    st = fresh_store()
    a = RNG.random((6, 224, 224), dtype=np.float32) + 0.1
    embed("S2B_43QHV_20240113_0_L2A", a, st)
    embed("S2A_43QHV_20240118_0_L2A", a, st)
    out = skills.compare_embeddings("emb_S2B_43QHV_20240113_0_L2A", "emb_S2A_43QHV_20240118_0_L2A", store=st)
    assert out["cosine_similarity"] == 1.0 and out["per_tile"]["min"] == 1.0


def test_rank_orders_by_similarity():
    st = fresh_store()
    ref = RNG.random((6, 224, 224), dtype=np.float32) + 0.1
    near = ref + 0.01 * RNG.random((6, 224, 224), dtype=np.float32)
    far = RNG.random((6, 224, 224), dtype=np.float32) * 3 + 0.1
    embed("S2B_43QHV_20240101_0_L2A", ref, st)
    embed("S2B_43QHV_20240102_0_L2A", far, st)
    embed("S2B_43QHV_20240103_0_L2A", near, st)
    out = skills.rank_similar(
        "emb_S2B_43QHV_20240101_0_L2A", ["emb_S2B_43QHV_20240102_0_L2A", "emb_S2B_43QHV_20240103_0_L2A"], store=st
    )
    assert [r["item_id"] for r in out["ranking"]][0] == "S2B_43QHV_20240103_0_L2A"
    sims = [r["cosine_similarity"] for r in out["ranking"]]
    assert sims == sorted(sims, reverse=True)


def test_unknown_id_and_unknown_skill_are_clear_errors():
    st = fresh_store()
    for call in (lambda: skills.compare_embeddings("nope", "nada", store=st), lambda: skills.run_skill("fly", {}),
                 lambda: skills.run_skill("embed_scene", {"bogus": 1})):
        try:
            call()
        except SkillError:
            continue
        raise AssertionError("expected SkillError")


def test_float_args_from_a2a_are_accepted():
    fresh_store()
    try:
        skills.run_skill("embed_scene", {"collection": "c", "item_id": "i", "patch_size": 100.0})
    except SkillError as exc:
        assert "multiple of 16" in str(exc)  # reached validation as an int, not a TypeError
    else:
        raise AssertionError("expected SkillError")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
