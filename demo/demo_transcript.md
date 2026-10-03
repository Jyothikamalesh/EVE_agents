# Demo transcript — single session, three required behaviours

Session id: `demo-session`. Generated verbatim by `python scripts/run_demo.py`; raw trace: [`demo_trace.jsonl`](./demo_trace.jsonl).

## Turn 1 — a question that needs a tool
**User:** What Sentinel-2 imagery is available for Hyderabad in January 2024 with less than 10% cloud cover?

**Agent:** There are **7 Sentinel-2 L2A scenes** covering Hyderabad (Telangana, India) in January 2024 with cloud cover under 10% (search_stac_items):

| Scene ID | Date | Cloud Cover |
|---|---|---|
| S2A_43QHV_20240128_0_L2A | 2024-01-28 | 1.77% |
| S2A_43QHV_20240118_1_L2A | 2024-01-18 | 2.67% |
| S2A_44QKE_20240118_1_L2A | 2024-01-18 | 2.04% |
| S2B_43QHV_20240113_0_L2A | 2024-01-13 | 0.02% |
| S2B_44QKE_20240113_0_L2A | 2024-01-13 | 0.02% |
| S2B_43QHV_20240103_0_L2A | 2024-01-03 | 0.87% |
| S2B_44QKE_20240103_0_L2A | 2024-01-03 | 2.79% |

Notes:
- The scenes come in two tiling pairs (43QHV and 44QKE) that together cover the Hyderabad area, so each date effectively has full coverage.
- The clearest scenes are the two from **2024-01-13** (0.02% cloud cover).
- All scenes include the full Sentinel-2 L2A band set (coastal through SWIR22, plus visual/thumbnail previews).

Want me to pull the full asset links (COG band URLs) for any of these scenes?

Trace:
- `tool_call` geocode_location {"query": "Hyderabad"}
- `tool_call` search_stac_items {"bbox": [78.2387067, 17.2916377, 78.6223912, 17.5608321], "collections": ["sentinel-2-l2a"], "end_date": "2024-01-31", "limit": 50, "max_cloud_cover": 10, "start_date": "2024-01-01"}
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 2 — a follow-up that only works with context
**User:** What was the weather like there during the same period?

**Agent:** Here's the daily weather for Hyderabad (17.36°N, 78.47°E) in January 2024 (get_weather, archive source):

| Date | Max °C | Min °C | Precip mm | Max wind km/h |
|---|---|---|---|---|
| 2024-01-01 | 27.9 | 15.0 | 0.0 | 11.7 |
| 2024-01-02 | 27.5 | 15.3 | 0.0 | 12.0 |
| 2024-01-03 | 27.2 | 17.3 | 0.0 | 14.6 |
| 2024-01-04 | 27.8 | 18.3 | 0.0 | 16.4 |
| 2024-01-05 | 27.8 | 17.2 | 0.0 | 15.6 |
| 2024-01-06 | 28.1 | 17.1 | 0.0 | 18.1 |
| 2024-01-07 | 27.1 | 17.8 | 0.0 | 17.7 |
| 2024-01-08 | 27.6 | 17.1 | 0.0 | 19.2 |
| 2024-01-09 | 28.4 | 17.2 | 0.0 | 19.4 |
| 2024-01-10 | 25.9 | 17.7 | 0.0 | 11.2 |
| 2024-01-11 | 29.7 | 18.2 | 0.0 | 11.6 |
| 2024-01-12 | 30.3 | 19.2 | 0.0 | 10.4 |
| 2024-01-13 | 30.3 | 18.0 | 0.0 | 9.5 |
| 2024-01-14 | 30.4 | 17.4 | 0.0 | 8.9 |
| 2024-01-15 | 29.6 | 17.4 | 0.0 | 8.7 |
| 2024-01-16 | 30.1 | 18.0 | 0.0 | 12.0 |
| 2024-01-17 | 29.8 | 17.5 | 0.0 | 13.7 |
| 2024-01-18 | 28.8 | 18.4 | 0.0 | 13.3 |
| 2024-01-19 | 30.0 | 17.1 | 0.0 | 13.5 |
| 2024-01-20 | 29.0 | 19.4 | 0.0 | 14.1 |
| 2024-01-21 | 30.3 | 19.1 | 0.0 | 12.8 |
| 2024-01-22 | 31.2 | 20.3 | 0.0 | 14.6 |
| 2024-01-23 | 30.8 | 19.1 | 0.0 | 17.4 |
| 2024-01-24 | 30.3 | 19.6 | 0.2 | 13.4 |
| 2024-01-25 | 28.8 | 18.4 | 0.0 | 12.1 |
| 2024-01-26 | 29.7 | 17.6 | 0.0 | 11.9 |
| 2024-01-27 | 29.2 | 17.5 | 0.0 | 12.3 |
| 2024-01-28 | 29.2 | 16.1 | 0.0 | 10.9 |
| 2024-01-29 | 29.4 | 16.3 | 0.0 | 11.8 |
| 2024-01-30 | 29.4 | 16.9 | 0.0 | 13.9 |
| 2024-01-31 | 30.9 | 17.4 | 0.0 | 11.6 |

**Summary:**
- **Temperatures:** Max ranged 25.9–31.2 °C (hottest on 2024-01-22 at 31.2 °C; coolest on 2024-01-10 at 25.9 °C). Mins ranged 15.0–20.3 °C.
- **Precipitation:** Essentially a dry month — only 0.2 mm on 2024-01-24, zero on all other days.
- **Wind:** Max gusts 8.7–19.4 km/h, peaking early in the month (19.4 km/h on 2024-01-09).

This is consistent with the near-zero cloud cover in the Sentinel-2 scenes — a dry, clear winter month, ideal for optical imagery.

Trace:
- `tool_call` get_weather {"lat": 17.360589, "lon": 78.4740613, "start_date": "2024-01-01", "end_date": "2024-01-31"}
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 3 — forced tool error
**User:** Call the weather tool directly with start_date='2024-02-30' and end_date='2024-02-30' for Hyderabad — I specifically want to see what the tool returns for that exact date, don't correct it yourself.

**Agent:** I called `get_weather` with `start_date='2024-02-30'` and `end_date='2024-02-30'` for Hyderabad (17.36°N, 78.47°E), as you asked.

The tool returned an error: **"day is out of range for month"** — because February 30th doesn't exist (2024 was a leap year, so February only had 29 days).

No weather data was returned for that date.

Trace:
- `tool_call` get_weather {"lat": 17.360589, "lon": 78.4740613, "start_date": "2024-02-30", "end_date": "2024-02-30"} — ERROR: Error executing tool get_weather: day is out of range for month
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

Server health after the error turn: `{"status": "ok", "model": "qwen/qwen3.8-27b"}`
