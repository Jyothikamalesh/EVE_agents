"""Offline tests for the groundedness checker. Run: python -m evals.test_groundedness"""

import json

from evals.groundedness import check_groundedness

SEARCH = json.dumps({
    "count": 2,
    "items": [
        {"id": "S2B_31UDQ_20240105_0_L2A", "datetime": "2024-01-05T10:50:21Z", "cloud_cover": 18.0104},
        {"id": "S2A_31UDQ_20240110_0_L2A", "datetime": "2024-01-10T10:50:21Z", "cloud_cover": 2.79},
    ],
})
WEATHER = json.dumps({"daily": {"temperature_2m_max": [24.0, 31.2, 28.4], "precipitation_sum": [0.0, 1.5, 0.0]}})
GEO = json.dumps({"best_match": {"display_name": "Paris, France", "lat": 48.8588897, "lon": 2.3200410}})


def ok(reply, outputs, user=()):
    return check_groundedness(reply, list(outputs), list(user))


def test_supported_values_pass():
    r = ok("2 scenes: S2B_31UDQ_20240105_0_L2A on 2024-01-05 (18.01% cloud) and 2.79% on 2024-01-10.", [SEARCH])
    assert r["grounded"], r


def test_rounding_and_derived_stats_pass():
    r = ok("Max 31.2°C, min 24, mean 27.9; total rain 1.5 mm. Paris is at 48.86, 2.32.", [WEATHER, GEO])
    assert r["grounded"], r


def test_invented_scene_and_date_flagged():
    r = ok("Best scene is S2A_99ZZZ_20240101_0_L2A on 2024-01-01.", [SEARCH])
    assert not r["grounded"]
    assert any("S2A_99ZZZ" in u for u in r["unsupported"]) and any("2024-01-01" in u for u in r["unsupported"])


def test_invented_number_flagged():
    r = ok("Cloud cover was 7.35% that day.", [SEARCH])
    assert not r["grounded"] and "number 7.35" in r["unsupported"]


def test_user_supplied_values_allowed():
    r = ok("Filtering below 10% cloud for 2024.", [SEARCH], user=["under 10% cloud in 2024"])
    assert r["grounded"], r


def test_no_tools_and_no_values_is_grounded():
    r = ok("I can only help with imagery, weather and place lookups.", [])
    assert r["grounded"] and r["checked"] == 0


def test_list_markers_ignored():
    r = ok("1. First scene\n2. Second scene", [SEARCH])
    assert r["grounded"], r


def test_day_of_month_derived_from_iso_datetime():
    r = ok("Scenes on Jan 5 and Jan 10.", [SEARCH])
    assert r["grounded"], r


def test_timestamp_checked_by_date_part():
    assert ok("Acquired 2024-01-05T10:50:21.000Z.", [SEARCH])["grounded"]
    assert not ok("Acquired 2024-01-07T10:50:21.000Z.", [SEARCH])["grounded"]


def test_ids_dates_mode_ignores_suggested_numbers_but_not_ids():
    assert check_groundedness("Try <=5% or <=10% cloud.", [SEARCH], [], mode="ids_dates")["grounded"]
    bad = check_groundedness("Try S2A_99ZZZ_20240101_0_L2A.", [SEARCH], [], mode="ids_dates")
    assert not bad["grounded"]


def test_allow_list():
    r = check_groundedness("Retry with 2024-02-29.", [], ["2024-02-30"], allow=["2024-02-29"])
    assert r["grounded"], r


def test_table_row_index_ignored():
    r = ok("| # | Scene |\n|---|---|\n| 1 | S2B_31UDQ_20240105_0_L2A |\n| 9 | S2A_31UDQ_20240110_0_L2A |", [SEARCH])
    assert r["grounded"], r


def test_unicode_minus_and_product_names():
    r = check_groundedness("Sentinel-2 over (\u22123.3811721, 37.3)", [json.dumps({"lat": -3.3811721, "lon": 37.3})], [])
    assert r["grounded"], r


def test_ratio_reported_as_percent():
    r = check_groundedness("Similarity is 73% (compare_embeddings)", [json.dumps({"cosine_similarity": 0.7338})], [])
    assert r["grounded"], r


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
