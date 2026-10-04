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


def test_python_repr_tool_output_gets_series_stats():
    """The tools node stores str(result): a Python repr, not JSON. Sums/min/max/mean must still work."""
    repr_weather = str({"daily": {"temperature_2m_max": [29.9, 31.2, 34.8], "precipitation_sum": [0.1, 1.5, 1.4, 0.6, 1.3, 0.8, 0.1]}})
    assert not repr_weather.startswith('{"')  # single quotes: json.loads would fail on this
    r = ok("Total rain was 5.8 mm and the hottest day reached 34.8 °C (get_weather).", [repr_weather])
    assert r["grounded"], r
    # a total that is not the sum of the series is still caught
    assert not ok("Total rain was 4.8 mm.", [repr_weather])["grounded"]


def test_hemisphere_notation_for_negative_coordinates():
    geo = str({"best_match": {"display_name": "Nairobi, Kenya", "lat": -1.2890006, "lon": 36.8172812}})
    assert ok("Nairobi is at 1.289°S, 36.817°E (geocode_location).", [geo])["grounded"]
    assert not ok("Nairobi is at 1.389°S.", [geo])["grounded"]   # a wrong magnitude is still caught


def test_capability_numbers_are_whole_numbers_from_tool_descriptions_only():
    cap = ["Forecast up to 16 days ahead; the archive goes back to 1940."]
    assert check_groundedness("The tool reaches 16 days ahead and back to 1940.", [], [], capability_texts=cap)["grounded"]
    assert not check_groundedness("The tool reaches 16 days ahead.", [], [])["grounded"]            # no description: unsupported
    assert not check_groundedness("It will be 16.5 degrees tomorrow.", [], [], capability_texts=cap)["grounded"]   # decimals need results


def test_advice_values_are_exempt_but_claims_and_ids_are_not():
    r = check_groundedness("No scenes found. Try <=10% or <=20% cloud, e.g. 2024-02-29.", [], [])
    assert r["grounded"], r
    assert check_groundedness("The cloud cover is 7.5%. Try <=10% instead.", [], [])["unsupported"] == ["number 7.5"]
    assert not check_groundedness("Try 10% cloud. The cloud cover is 10%.", [], [])["grounded"]      # also in a plain claim
    assert not check_groundedness("For example S2A_99ZZZ_20240101_0_L2A looks clear.", [], [])["grounded"]   # ids never exempt
    assert not check_groundedness("Rain on 2024-03-05 was heavy.", [], [])["grounded"]               # a plain dated claim


def test_tool_citation_needs_a_value():
    from agents.graphs.grounding import find_uncalled_tool_citations as cites
    names = ["get_weather", "search_stac_items"]
    assert cites("Max 31.2 (get_weather).", names, []) == ["tool get_weather cited but never called"]
    assert cites("Daily weather for Hyderabad in 2024 (get_weather, archive source):", names, []) == ["tool get_weather cited but never called"]
    assert cites("The tool (get_weather) only reaches 16 days.", names, []) == []
    assert cites("I could use `get_weather` or search_stac_items for that.", names, []) == []
    assert cites("Max 31.2 (get_weather).", names, ["get_weather"]) == []


if __name__ == "__main__":
    fns =[v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
