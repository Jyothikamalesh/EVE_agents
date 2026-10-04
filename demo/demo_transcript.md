# Demo transcript — single session, three required behaviours

Session id: `demo-1791125933`. Generated verbatim by `python scripts/run_demo.py`; raw trace: [`demo_trace.jsonl`](./demo_trace.jsonl).

## Turn 1 — a question that needs a tool
**User:** What Sentinel-2 imagery is available for Hyderabad in January 2024 with less than 10% cloud cover?

**Agent:** There are **7 Sentinel-2 L2A scenes** covering Hyderabad in January 2024 with cloud cover under 10% (search_stac_items):

| Scene ID | Date | Cloud Cover |
|---|---|---|
| S2A_43QHV_20240128_0_L2A | 2024-01-28 | 1.77% |
| S2A_43QHV_20240118_1_L2A | 2024-01-18 | 2.67% |
| S2A_44QKE_20240118_1_L2A | 2024-01-18 | 2.04% |
| S2B_43QHV_20240113_0_L2A | 2024-01-13 | 0.02% |
| S2B_44QKE_20240113_0_L2A | 2024-01-13 | 0.02% |
| S2B_43QHV_20240103_0_L2A | 2024-01-03 | 0.87% |
| S2B_44QKE_20240103_0_L2A | 2024-01-03 | 2.79% |

The scenes come in pairs from two adjacent MGRS tiles (43QHV and 44QKE) that together cover the Hyderabad area. The clearest acquisitions are the January 13 pair, both with essentially zero cloud cover (~0.02%).

Would you like me to pull full metadata or asset links for any of these scenes?

Trace:
- `tool_call` geocode_location {"query": "Hyderabad"}
- `tool_call` search_stac_items {"bbox": [78.2387067, 17.2916377, 78.6223912, 17.5608321], "collections": ["sentinel-2-l2a"], "end_date": "2024-01-31", "limit": 50, "max_cloud_cover": 10, "start_date": "2024-01-01"}
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 2 — a follow-up that only works with context
**User:** What was the weather like there during the same period?

**Agent:** Here's the weather in Hyderabad for January 2024 (get_weather, archive data):

**Overall:** A dry, warm month — typical of the local winter. Max temperatures ranged from 25.9 °C (Jan 10) to 31.2 °C (Jan 22), with most days between 27–30 °C. Mins stayed in the 15–20 °C range.

**Precipitation:** Essentially none — the only measurable rain was 0.2 mm on Jan 24. Every other day recorded 0.0 mm.

**Wind:** Light to moderate, with daily max gusts mostly 9–19 km/h; the windiest stretch was Jan 6–9 (17.7–19.4 km/h).

**On the imagery dates specifically:**
- Jan 3: max 27.2 °C, no rain, max wind 14.6 km/h
- Jan 13: max 30.3 °C, no rain, max wind 9.5 km/h
- Jan 18: max 28.8 °C, no rain, max wind 13.3 km/h
- Jan 28: max 29.2 °C, no rain, max wind 10.9 km/h

The dry, cloud-free conditions on those days are consistent with the very low cloud cover (0.02–2.79%) reported for the Sentinel-2 scenes.

Trace:
- `tool_call` get_weather {"end_date": "2024-01-31", "lat": 17.360589, "lon": 78.4740613, "start_date": "2024-01-01"}
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 3 — forced tool error
**User:** Call the weather tool directly with start_date='2024-02-30' and end_date='2024-02-30' for Hyderabad — I specifically want to see what the tool returns for that exact date, don't correct it yourself.

**Agent:** I called get_weather with exactly `start_date='2024-02-30'` and `end_date='2024-02-30'` for Hyderabad (17.360589, 78.4740613), as you asked.

The tool returned an error:

> **Error executing tool get_weather: day is out of range for month**

So the tool rejects the date outright — February 30th doesn't exist, and the underlying date parser raises a "day is out of range for month" error rather than clamping or correcting it. No weather data was returned.

If you'd like, I can pull the actual weather for a valid date in February 2024 instead.

Trace:
- `tool_call` get_weather {"end_date": "2024-02-30", "lat": 17.360589, "lon": 78.4740613, "start_date": "2024-02-30"} — ERROR: Error executing tool get_weather: day is out of range for month
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

Server health after the error turn: `{"status": "ok", "model": "groq:qwen/qwen3.8-27b", "tools": ["geocode_location", "list_stac_collections", "search_stac_items", "get_stac_item", "get_weather", "embed_scene", "compare_embeddings", "rank_similar"]}`
