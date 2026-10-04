# TerraMind A2A demo

Session `terramind-demo-1791127141`. The EO agent's `embed_scene`, `compare_embeddings` and `rank_similar` tools are A2A calls to the separate TerraMind agent (`terramind_agent/`); the (196, 192) tensors stay in that agent, only embedding ids and summary numbers reach the LLM. Generated verbatim by `python scripts/run_demo_terramind.py`.

## Turn 1
**User:** Find three Sentinel-2 scenes over Hyderabad in January 2024 with under 5% cloud, all on the same MGRS tile, and embed each of them with TerraMind.

**Agent:** Done. I found three Sentinel-2 L2A scenes over Hyderabad in January 2024, all on MGRS tile **43QHV** with under 5% cloud, and embedded each with TerraMind:

| Scene | Date | Cloud cover | Embedding ID |
|---|---|---|---|
| S2B_43QHV_20240113_0_L2A | 2024-01-13 | 0.02% (search_stac_items) | `emb_S2B_43QHV_20240113_0_L2A` (embed_scene) |
| S2A_43QHV_20240118_1_L2A | 2024-01-18 | 2.67% (search_stac_items) | `emb_S2A_43QHV_20240118_1_L2A` (embed_scene) |
| S2A_43QHV_20240128_0_L2A | 2024-01-28 | 1.77% (search_stac_items) | `emb_S2A_43QHV_20240128_0_L2A` (embed_scene) |

Notes:
- Each embedding is a (196, 192) tensor from a 224 px centred crop (a ~2.2 km square at the centre of the 43QHV tile, crop_bbox [78.33187, 17.5543, 78.35332, 17.57487]) — not necessarily over central Hyderabad itself (embed_scene).
- The full tensors are stored server-side (embed_scene).

Trace:
- `tool_call` geocode_location {"query": "Hyderabad"} (86.7 ms)
- `tool_call` search_stac_items {"bbox": [78.2387067, 17.2916377, 78.6223912, 17.5608321], "collections": ["sentinel-2-l2a"], "end_date": "2024-01-31", "limit": 50, "max_cloud_cover": 5, "start_date": "2024-01-01"} (1442.8 ms)
- `tool_call` embed_scene {"collection": "sentinel-2-l2a", "item_id": "S2A_43QHV_20240128_0_L2A", "patch_size": 224} (33218.6 ms)
- `tool_call` embed_scene {"collection": "sentinel-2-l2a", "item_id": "S2A_43QHV_20240118_1_L2A", "patch_size": 224} (26628.8 ms)
- `tool_call` embed_scene {"collection": "sentinel-2-l2a", "item_id": "S2B_43QHV_20240113_0_L2A", "patch_size": 224} (26406.8 ms)
- `verify`  {"issues": ["tool compare_embeddings cited but never called", "tool rank_similar cited but never called"], "rewritten": true, "caveat": false}

## Turn 2
**User:** Rank the other two by how similar they are to the first one.

**Agent:** Ranked against the reference scene **S2B_43QHV_20240113_0_L2A** (2024-01-13, 0.02% cloud):

1. **S2A_43QHV_20240128_0_L2A** (2024-01-28, 1.77% cloud, 15 days apart) — cosine similarity **0.9989** (rank_similar)
2. **S2A_43QHV_20240118_1_L2A** (2024-01-18, 2.67% cloud, 5 days apart) — cosine similarity **0.9982** (rank_similar)

How to read this:
- These scores are compressed near 1.0 — for the same tile on different dates (days to weeks apart), typical values are 0.998–0.999, so both scenes look very close to the reference (rank_similar reference_points).
- The difference between 0.9989 and 0.9982 is only in the third decimal place, which is not meaningful — treat them as **tied** (rank_similar reading_guide).
- Both share the same footprint (same 43QHV tile), so the per-tile comparison applies: mean per-tile similarity is 0.9863 and 0.9869 respectively.
- These are appearance embeddings with no land-cover labels, so I can't say what (if anything) changed — only that the two scenes are essentially equally similar to the 13 January reference.

Trace:
- `tool_call` rank_similar {"reference_id": "emb_S2B_43QHV_20240113_0_L2A", "candidate_ids": ["emb_S2A_43QHV_20240118_1_L2A", "emb_S2A_43QHV_20240128_0_L2A"]} (14.7 ms)
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 3
**User:** Where do the first scene and the least similar one differ most? Is the per-tile comparison valid for them?

**Agent:** Yes — the per-tile comparison is valid here: both scenes are on the same MGRS tile (43QHV), so `same_footprint` is true and the per-tile grid applies (compare_embeddings).

**Where they differ most** (compare_embeddings, 14×14 grid, 0-based from top-left):

| Grid cell (row, col) | Per-tile similarity |
|---|---|
| (6, 1) | 0.8322 — lowest |
| (5, 5) | 0.8529 |
| (5, 4) | 0.8803 |

Overall context:
- Overall cosine similarity: 0.9982; per-tile mean 0.9869, min 0.8322, max 0.9981 (compare_embeddings).
- The scenes are 5 days apart (2024-01-13 vs 2024-01-18), with cloud cover 0.02% vs 2.67% (compare_embeddings).

Caveats (per the tool's reading guide):
- A low per-tile score only says the **appearance** changed in that grid cell — it doesn't say what changed. Cloud, shadow, sun angle, or seasonal effects can all cause it; the 2.67% cloud in the 18 January scene is a plausible contributor.
- These are appearance embeddings with no land-cover labels, so I can't name a type of change (e.g. deforestation or flooding).

Trace:
- `tool_call` compare_embeddings {"id_a": "emb_S2B_43QHV_20240113_0_L2A", "id_b": "emb_S2A_43QHV_20240118_1_L2A"} (22.2 ms)
- `verify`  {"issues": [], "rewritten": false, "caveat": false}
