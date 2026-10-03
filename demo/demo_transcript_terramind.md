# TerraMind A2A demo

Session `terramind-demo-1791052792`. The EO agent's `embed_scene`, `compare_embeddings` and `rank_similar` tools are A2A calls to the separate TerraMind agent (`terramind_agent/`); the (196, 192) tensors stay in that agent, only embedding ids and summary numbers reach the LLM. Generated verbatim by `python scripts/run_demo_terramind.py`.

## Turn 1
**User:** Find three Sentinel-2 scenes over Hyderabad in January 2024 with under 5% cloud, all on the same MGRS tile, and embed each of them with TerraMind.

**Agent:** Found three Sentinel-2 L2A scenes over Hyderabad in January 2024, all on MGRS tile **43QHV** with under 5% cloud, and embedded each with TerraMind:

| Scene | Date | Cloud cover | Embedding ID |
|---|---|---|---|
| S2B_43QHV_20240113_0_L2A | 2024-01-13 | 0.02% | `emb_S2B_43QHV_20240113_0_L2A` |
| S2A_43QHV_20240118_1_L2A | 2024-01-18 | 2.67% | `emb_S2A_43QHV_20240118_1_L2A` |
| S2A_43QHV_20240128_0_L2A | 2024-01-28 | 1.77% | `emb_S2A_43QHV_20240128_0_L2A` |

All three share the same footprint (tile 43QHV, crop bbox 78.33187, 17.5543 → 78.35332, 17.57487), so per-tile change comparison is available. Want me to compare them (e.g. 13 Jan vs 28 Jan) or rank them against a reference?

Trace:
- `tool_call` geocode_location {"query": "Hyderabad"} (108.6 ms)
- `tool_call` search_stac_items {"bbox": [78.2387067, 17.2916377, 78.6223912, 17.5608321], "collections": ["sentinel-2-l2a"], "end_date": "2024-01-31", "limit": 50, "max_cloud_cover": 5, "start_date": "2024-01-01"} (2323.5 ms)
- `tool_call` embed_scene {"collection": "sentinel-2-l2a", "item_id": "S2A_43QHV_20240128_0_L2A", "patch_size": 224} (2612.4 ms)
- `tool_call` embed_scene {"collection": "sentinel-2-l2a", "item_id": "S2A_43QHV_20240118_1_L2A", "patch_size": 224} (27939.3 ms)
- `tool_call` embed_scene {"collection": "sentinel-2-l2a", "item_id": "S2B_43QHV_20240113_0_L2A", "patch_size": 224} (26650.4 ms)
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 2
**User:** Rank the other two by how similar they are to the first one.

**Agent:** Ranked by similarity to the reference scene **S2B_43QHV_20240113_0_L2A** (2024-01-13, 0.02% cloud):

1. **S2A_43QHV_20240128_0_L2A** (2024-01-28, 1.77% cloud) — cosine similarity **0.9989**, mean per-tile similarity 0.9863
2. **S2A_43QHV_20240118_1_L2A** (2024-01-18, 2.67% cloud) — cosine similarity **0.9982**, mean per-tile similarity 0.9869

Both scenes share the same footprint (tile 43QHV), so the per-tile comparison applies. The 28 Jan scene is marginally closer to the 13 Jan reference overall, though the 18 Jan scene edges it out slightly on mean per-tile similarity — the differences are very small, indicating little change over the tile across January 2024.

Trace:
- `tool_call` rank_similar {"reference_id": "emb_S2B_43QHV_20240113_0_L2A", "candidate_ids": ["emb_S2A_43QHV_20240118_1_L2A", "emb_S2A_43QHV_20240128_0_L2A"]} (18.1 ms)
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 3
**User:** Where do the first scene and the least similar one differ most? Is the per-tile comparison valid for them?

**Agent:** Yes — the per-tile comparison is valid: both scenes share tile id **43QHV** (`same_footprint: true`), so the grid-based difference map applies.

Comparing **S2B_43QHV_20240113_0_L2A** (2024-01-13) vs **S2A_43QHV_20240118_1_L2A** (2024-01-18, the least similar of the two):

- Overall cosine similarity: **0.9982**
- Per-tile (14×14 grid): mean **0.9869**, min **0.8322**, max **0.9981**

The three least-similar grid cells (0-based, from top-left):

| Rank | Row | Col | Tile similarity |
|---|---|---|---|
| 1 | 6 | 1 | 0.8322 |
| 2 | 5 | 5 | 0.8529 |
| 3 | 5 | 4 | 0.8803 |

So the biggest differences between the two dates are concentrated in the upper-middle-left area of the tile (rows 5–6, cols 1 and 4–5), with the single most divergent cell at row 6, col 1 (similarity 0.8322). The rest of the tile is nearly identical (up to 0.9981).

Trace:
- `tool_call` compare_embeddings {"id_a": "emb_S2B_43QHV_20240113_0_L2A", "id_b": "emb_S2A_43QHV_20240118_1_L2A"} (16.0 ms)
- `verify`  {"issues": [], "rewritten": false, "caveat": false}
