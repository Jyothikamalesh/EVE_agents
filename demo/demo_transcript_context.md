# Context-policy demo (session `ctx-demo-1791126214`)

## Turn 1
**User:** What Sentinel-2 imagery is available for Hyderabad in January 2024 with less than 10% cloud cover?

**Reply:** There are **7 Sentinel-2 L2A scenes** covering Hyderabad in January 2024 with cloud cover under 10% (search_stac_items):

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

## Turn 2
**User:** What was the weather like there during the same period?

**Reply:** Here's the weather in Hyderabad for January 2024 (get_weather, archive data):

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

## Turn 3
**User:** Where is Nairobi? Give me its coordinates.

**Reply:** Nairobi, Kenya is located at:

- **Latitude:** -1.3024° (1.30° S)
- **Longitude:** 36.8289° (36.83° E)

Its bounding box is [36.6647, -1.4449, 37.1049, -1.1607] (geocode_location).

Trace:
- `tool_call` geocode_location {"query": "Nairobi"}
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 4
**User:** What was the weather in Nairobi from 2023-07-01 to 2023-07-07?

**Reply:** Here's the weather in Nairobi from July 1–7, 2023 (get_weather, archive data):

| Date | Max Temp | Min Temp | Precipitation | Max Wind |
|---|---|---|---|---|
| Jul 1 | 21.9 °C | 13.6 °C | 0.6 mm | 12.6 km/h |
| Jul 2 | 21.2 °C | 14.7 °C | 0.3 mm | 9.9 km/h |
| Jul 3 | 21.2 °C | 13.1 °C | 0.6 mm | 8.3 km/h |
| Jul 4 | 20.5 °C | 14.0 °C | 2.0 mm | 10.9 km/h |
| Jul 5 | 23.2 °C | 11.5 °C | 0.0 mm | 8.9 km/h |
| Jul 6 | 21.6 °C | 11.4 °C | 0.1 mm | 12.4 km/h |
| Jul 7 | 20.2 °C | 13.3 °C | 0.0 mm | 12.0 km/h |

**Summary:** A mild, mostly dry week. Max temperatures hovered around 20–23 °C (warmest on Jul 5 at 23.2 °C), with cool nights dipping to 11.4 °C (Jul 6). Rain was light and sporadic — the only notable amount was 2.0 mm on Jul 4, with Jul 5 and Jul 7 completely dry. Winds were light, peaking at 12.6 km/h on Jul 1.

Trace:
- `tool_call` get_weather {"end_date": "2023-07-07", "lat": -1.302398, "lon": 36.8288509, "start_date": "2023-07-01"}
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 5
**User:** What imagery collections can you search?

**Reply:** Here are the imagery collections I can search (list_stac_collections):

**Optical (cloud-filterable):**
- **sentinel-2-l2a** — Sentinel-2 Level-2A (atmospherically corrected, MSI)
- **sentinel-2-l1c** — Sentinel-2 Level-1C (top-of-atmosphere)
- **sentinel-2-c1-l2a** — Sentinel-2 Collection 1 Level-2A
- **sentinel-2-pre-c1-l2a** — Sentinel-2 Pre-Collection 1 Level-2A (baseline < 05.00)
- **landsat-c2-l2** — Landsat Collection 2 Level-2 (Landsat 4/5 TM, 7 ETM+, 8/9 OLI+TIRS)
- **naip** — NAIP: National Agriculture Imagery Program (U.S.-wide, high-res aerial, R/G/B/IR, 2010–present)

**Radar (cloud-penetrating):**
- **sentinel-1-grd** — Sentinel-1 Level-1C Ground Range Detected (SAR, 6-day revisit, works through clouds)

**Elevation / Digital Surface Models:**
- **cop-dem-glo-30** — Copernicus DEM GLO-30 (30 m, limited worldwide coverage)
- **cop-dem-glo-90** — Copernicus DEM GLO-90 (90 m, worldwide)

The default for searches is **sentinel-2-l2a**. Want me to search any of these for a location and date range?

Trace:
- `tool_call` list_stac_collections {}
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 6
**User:** Find Sentinel-2 scenes over Nairobi in March 2024 under 30% cloud, limit 3.

**Reply:** Found 3 Sentinel-2 L2A scenes over Nairobi in March 2024 with <30% cloud cover (search_stac_items):

| Scene ID | Date | Cloud Cover | Tile |
|---|---|---|---|
| **S2B_36MZD_20240318_0_L2A** | 2024-03-18 | 26.75% | 36MZD |
| **S2B_37MBU_20240318_0_L2A** | 2024-03-18 | 26.24% | 37MBU |
| **S2A_36MZD_20240313_0_L2A** | 2024-03-13 | 2.39% | 36MZD |

The clearest scene is **S2A_36MZD_20240313_0_L2A** with only 2.39% cloud cover. The two March 18 scenes have higher cloud (~26%) but still under your 30% threshold.

Want me to pull full asset links for any of these, or search a wider date range?

Trace:
- `tool_call` search_stac_items {"bbox": [36.6647016, 1.4448822, 37.1048735, 1.1606749], "collections": ["sentinel-2-l2a"], "end_date": "2024-03-31", "limit": 3, "max_cloud_cover": 30, "start_date": "2024-03-01"}
- `tool_call` search_stac_items {"bbox": [36.6647016, -1.4448822, 37.1048735, -1.1606749], "collections": ["sentinel-2-l2a"], "end_date": "2024-03-31", "limit": 3, "max_cloud_cover": 30, "start_date": "2024-03-01"}
- `context_summary`  - **Hyderabad, Telangana, India**: lat 17.360589, lon 78.4740613; bbox [78.2387067, 17.2916377, 78.6223912, 17.5608321].
- **Sentinel-2 L2A search (Jan 2024, <10% cloud)**: 7 scenes found.
  - S2A_43QHV_20240128_0_L2A (2024-01-28, 1.77% cloud)
  - S2A_43QHV_20240118_1_L2A (2024-01-18, 2.67% cloud)
  - S2A_44QKE_20240118_1_L2A (2024-01-18, 2.04% cloud)
  - S2B_43QHV_20240113_0_L2A (2024-01-13, 0.02% cloud)
  - S2B_44QKE_20240113_0_L2A (2024-01-13, 0.02% cloud)
  - S2B_43QHV_20240103_0_L2A (2024-01-03, 0.87% cloud)
  - S2B_44QKE_20240103_0_L2A (2024-01-03, 2.79% cloud)
- **Hyderabad Weather (Jan 2024)**:
  - Max temp: 25.9°C (Jan 10) to 31.2°C (Jan 22); mean 29.11°C.
  - Min temp: 15.0°C (Jan 1) to 20.3°C (Jan 22); mean 17.71°C.
  - Precipitation: Total 0.2 mm (only on Jan 24); all other days 0.0 mm.
  - Wind: Max gusts 8.7–19.4 km/h; mean 13.36 km/h.
- **Nairobi, Kenya**: lat -1.302398, lon 36.8288509; bbox [36.6647016, -1.4448822, 37.1048735, -1.1606749].
- `verify`  {"issues": [], "rewritten": false, "caveat": false}

## Turn 7
**User:** Going back to the very first thing I asked: which city was it, what cloud-cover limit did I set, and what date range?

**Reply:** Your very first question was about **Hyderabad** (Telangana, India), asking for **Sentinel-2 imagery** with **less than 10% cloud cover** in **January 2024**.

That search returned 7 scenes, the clearest being S2B_43QHV_20240113_0_L2A and S2B_44QKE_20240113_0_L2A (both 0.02% cloud on 2024-01-13).

Trace:
- `verify`  {"issues": [], "rewritten": false, "caveat": false}
