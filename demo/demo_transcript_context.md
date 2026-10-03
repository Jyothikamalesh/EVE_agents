# Context-policy demo (session `ctx-demo-1791050346`)

## Turn 1
**User:** What Sentinel-2 imagery is available for Hyderabad in January 2024 with less than 10% cloud cover?

**Reply:** There are **7 Sentinel-2 L2A scenes** covering Hyderabad (Telangana, India) in January 2024 with cloud cover under 10% (search_stac_items):

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
- All scenes are atmospherically corrected L2A with full band assets (visual, NIR, red-edge, SWIR, etc.) available as COGs.

Want me to pull the full asset links for any specific scene?

Trace:
- `tool_call` geocode_location {"query": "Hyderabad"}
- `tool_call` search_stac_items {"bbox": [78.2387067, 17.2916377, 78.6223912, 17.5608321], "collections": ["sentinel-2-l2a"], "end_date": "2024-01-31", "limit": 50, "max_cloud_cover": 10, "start_date": "2024-01-01"}
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 2
**User:** What was the weather like there during the same period?

**Reply:** Here's the daily weather for Hyderabad (17.36°N, 78.47°E) in January 2024 (get_weather, archive source):

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
- **Dry month** — essentially no rain all month; the only measurable precipitation was 0.2 mm on 2024-01-24.
- **Temperatures** typical of the dry season: maxes ranged 25.9–31.2 °C (hottest day 2024-01-22 at 31.2 °C), mins 15.0–20.3 °C.
- **Winds** light to moderate, peaking at 19.4 km/h on 2024-01-09.

This is consistent with the very low cloud cover in the Sentinel-2 scenes — the near-zero cloud days (e.g., 0.02% on 2024-01-13) line up with the dry, clear conditions.

Trace:
- `tool_call` get_weather {"end_date": "2024-01-31", "lat": 17.360589, "lon": 78.4740613, "start_date": "2024-01-01"}
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 3
**User:** Where is Nairobi? Give me its coordinates.

**Reply:** Nairobi is a city in Kenya (geocode_location).

- **Coordinates:** Latitude −1.2890006, Longitude 36.8172812 (geocode_location)
- **Bounding box:** [36.6647016, −1.4448822, 37.1048735, −1.1606749] (geocode_location)

Trace:
- `tool_call` geocode_location {"query": "Nairobi"}
- `verify`  {"issues": ["number 1.289"], "rewritten": true, "caveat": false}

## Turn 4
**User:** What was the weather in Nairobi from 2023-07-01 to 2023-07-07?

**Reply:** Here's the daily weather for Nairobi (−1.3005272, 36.824646) from 2023-07-01 to 2023-07-07 (get_weather, archive source):

| Date | Max °C | Min °C | Precip mm | Max wind km/h |
|---|---|---|---|---|
| 2023-07-01 | 21.9 | 13.6 | 0.6 | 12.6 |
| 2023-07-02 | 21.2 | 14.7 | 0.3 | 9.9 |
| 2023-07-03 | 21.2 | 13.1 | 0.6 | 8.3 |
| 2023-07-04 | 20.5 | 14.0 | 2.0 | 10.9 |
| 2023-07-05 | 23.2 | 11.5 | 0.0 | 8.9 |
| 2023-07-06 | 21.6 | 11.4 | 0.1 | 12.4 |
| 2023-07-07 | 20.2 | 13.3 | 0.0 | 12.0 |

**Summary:**
- **Temperatures:** maxes 20.2–23.2 °C (warmest 2023-07-05 at 23.2 °C), mins 11.4–14.7 °C (coolest 2023-07-06 at 11.4 °C).
- **Precipitation:** mostly dry, with the highest daily amount on 2023-07-04 (2.0 mm).
- **Winds:** light, peaking at 12.6 km/h on 2023-07-01.

Trace:
- `tool_call` get_weather {"end_date": "2023-07-07", "lat": -1.2890006, "lon": 36.8172812, "start_date": "2023-07-01"}
- `verify`  {"issues": ["number 3.6"], "rewritten": true, "caveat": false}

## Turn 5
**User:** What imagery collections can you search?

**Reply:** Here are the imagery collections available to search (list_stac_collections, catalog: earth-search.aws.element84.com):

| Collection ID | Title | Description |
|---|---|---|
| `sentinel-2-l2a` | Sentinel-2 Level-2A | Global Sentinel-2 MSI data, atmospherically corrected (the one we used for Hyderabad) |
| `sentinel-2-l1c` | Sentinel-2 Level-1C | Global Sentinel-2 MSI data, top-of-atmosphere |
| `sentinel-2-c1-l2a` | Sentinel-2 Collection 1 Level-2A | Sentinel-2 Collection 1 L2A |
| `sentinel-2-pre-c1-l2a` | Sentinel-2 Pre-Collection 1 Level-2A | Pre-Collection 1 L2A (baseline < 05.00) |
| `sentinel-1-grd` | Sentinel-1 Level-1C GRD | SAR (radar) imagery — works through clouds, 6-day revisit |
| `landsat-c2-l2` | Landsat Collection 2 Level-2 | Atmospherically corrected Landsat 4/5/7/8/9 (TM, ETM+, OLI/TIRS) |
| `naip` | NAIP: National Agriculture Imagery Program | U.S.-wide high-res aerial imagery (R, G, B, IR), 2010–present |
| `cop-dem-glo-30` | Copernicus DEM GLO-30 | Digital Surface Model, 30 m, limited worldwide coverage |
| `cop-dem-glo-90` | Copernicus DEM GLO-90 | Digital Surface Model, 90 m, worldwide |

So you can search optical (Sentinel-2, Landsat, NAIP), radar (Sentinel-1), and elevation (Copernicus DEM) data. Want me to search any of these for a location or date range?

Trace:
- `tool_call` list_stac_collections {}
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 6
**User:** Find Sentinel-2 scenes over Nairobi in March 2024 under 30% cloud, limit 3.

**Reply:** Found 3 Sentinel-2 L2A scenes over Nairobi in March 2024, all well under 30% cloud (search_stac_items):

| Scene ID | Date | Cloud cover |
|---|---|---|
| `S2A_36NZG_20240323_0_L2A` | 2024-03-23 | 2.20% |
| `S2A_37NBB_20240323_0_L2A` | 2024-03-23 | 2.66% |
| `S2B_36NZG_20240321_0_L2A` | 2024-03-21 | 0.36% |

All three are atmospherically corrected L2A with full band assets (visual, NIR, red-edge, SWIR, etc.). The clearest is `S2B_36NZG_20240321_0_L2A` at 0.36% cloud. Want me to pull full metadata/asset links for any of these?

Trace:
- `tool_call` search_stac_items {"bbox": [36.6647016, 1.4448822, 37.1048735, 1.1606749], "collections": ["sentinel-2-l2a"], "end_date": "2024-03-31", "limit": 3, "max_cloud_cover": 30, "start_date": "2024-03-01"}
- `tool_call` search_stac_items {"bbox": [36.6647016, 1.1606749, 37.1048735, 1.4448822], "collections": ["sentinel-2-l2a"], "end_date": "2024-03-31", "limit": 3, "max_cloud_cover": 30, "start_date": "2024-03-01"}
- `context_summary`  - **Hyderabad, Telangana, India**
  - Coordinates: 17.360589°N, 78.4740613°E
  - Bounding box: [78.2387067, 17.2916377, 78.6223912, 17.5608321]
  - **Sentinel-2 L2A Imagery (Jan 2024, <10% cloud):** 7 scenes found.
    - 2024-01-28: S2A_43QHV_20240128_0_L2A (1.77%)
    - 2024-01-18: S2A_43QHV_20240118_1_L2A (2.67%), S2A_44QKE_20240118_1_L2A (2.04%)
    - 2024-01-13: S2B_43QHV_20240113_0_L2A (0.02%), S2B_44QKE_20240113_0_L2A (0.02%)
    - 2024-01-03: S2B_43QHV_20240103_0_L2A (0.87%), S2B_44QKE_20240103_0_L2A (2.79%)
  - **Weather (Jan 2024):** Dry month; total precipitation 0.2 mm (only on 2024-01-24). Max temps 25.9–31.2°C (peak 31.2°C on 2024-01-22); Min temps 15.0–20.3°C. Max wind speed 19.4 km/h (2024-01-09).
- **Nairobi, Kenya**
  - Coordinates: -1.2890006°N, 36.8172812°E
  - Bounding box: [36.6647016, -1.4448822, 37.1048735, -1.1606749]
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 7
**User:** Going back to the very first thing I asked: which city was it, what cloud-cover limit did I set, and what date range?

**Reply:** Your very first request was for **Hyderabad, Telangana, India** (17.360589°N, 78.4740613°E), with a **cloud-cover limit of under 10%**, for **January 2024** (2024-01-01 to 2024-01-31).

That search returned 7 Sentinel-2 L2A scenes, the clearest being the 2024-01-13 pair (`S2B_43QHV_20240113_0_L2A` and `S2B_44QKE_20240113_0_L2A`) at just 0.02% cloud.

Trace:
- `verify`  {"issues": [], "rewritten": false, "caveat": false}
