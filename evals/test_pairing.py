"""Offline tests for record-level pairing and superlative checks. Run: python -m evals.test_pairing"""

from evals.groundedness import check_groundedness

# tool results are stored as str(result): a Python repr, not JSON
SEARCH = str({
    "count": 3,
    "items": [
        {"id": "S2C_43PGQ_20260328_0_L2A", "datetime": "2026-03-28T05:11:00Z", "cloud_cover": 4.66, "bbox": [77.0, 12.0, 78.0, 13.0]},
        {"id": "S2C_43PHQ_20260328_0_L2A", "datetime": "2026-03-28T05:11:00Z", "cloud_cover": 1.16, "bbox": [77.0, 12.0, 78.0, 13.0]},
        {"id": "S2B_43PGQ_20260323_0_L2A", "datetime": "2026-03-23T05:11:00Z", "cloud_cover": 2.64, "bbox": [77.0, 12.0, 78.0, 13.0]},
    ],
})
WEATHER = str({
    "daily": {
        "time": ["2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12"],
        "temperature_2m_max": [30.1, 29.9, 34.8, 31.2],
        "temperature_2m_min": [19.4, 20.3, 21.7, 20.9],
        "precipitation_sum": [0.0, 1.5, 0.0, 0.8],
        "windspeed_10m_max": [11.7, 8.7, 19.4, 14.6],
    }
})
RANKING = str({
    "reference_id": "emb_S2B_43QHV_20240113_0_L2A",
    "ranking": [
        {"embedding_id": "emb_S2A_43QHV_20240128_0_L2A", "item_id": "S2A_43QHV_20240128_0_L2A", "datetime": "2024-01-28T05:00:00Z",
         "cloud_cover": 1.77, "cosine_similarity": 0.9991, "mean_tile_similarity": 0.9807},
        {"embedding_id": "emb_S2A_43QHV_20240118_1_L2A", "item_id": "S2A_43QHV_20240118_1_L2A", "datetime": "2024-01-18T05:00:00Z",
         "cloud_cover": 2.67, "cosine_similarity": 0.9982, "mean_tile_similarity": 0.9869},
    ],
})


def ok(reply, *outputs):
    return check_groundedness(reply, list(outputs), [])


def issues(reply, *outputs):
    return [u for u in ok(reply, *outputs)["unsupported"] if u.startswith("pairing")]


# ── scenes ───────────────────────────────────────────────────────────────────

def test_correct_list_and_table_pass():
    assert ok("1. S2C_43PGQ_20260328_0_L2A: 2026-03-28, cloud 4.66%\n2. S2C_43PHQ_20260328_0_L2A: 2026-03-28, cloud 1.16%", SEARCH)["grounded"]
    table = "| Scene | Date | Cloud |\n|---|---|---|\n| S2C_43PGQ_20260328_0_L2A | 2026-03-28 | 4.66% |\n| S2B_43PGQ_20260323_0_L2A | 2026-03-23 | 2.64% |"
    assert ok(table, SEARCH)["grounded"]


def test_number_on_the_wrong_scene_is_flagged():
    got = issues("1. S2C_43PGQ_20260328_0_L2A: cloud 1.16%\n2. S2C_43PHQ_20260328_0_L2A: cloud 4.66%", SEARCH)
    assert len(got) == 2 and "belongs to S2C_43PHQ_20260328_0_L2A" in got[0], got


def test_date_on_the_wrong_scene_is_flagged():
    got = issues("S2B_43PGQ_20260323_0_L2A was acquired on 2026-03-28 (2.64% cloud).", SEARCH)
    assert got == ["pairing 2026-03-28: not the date of S2B_43PGQ_20260323_0_L2A"], got


def test_false_superlative_flagged_true_one_passes():
    assert ok("The clearest scene is S2C_43PHQ_20260328_0_L2A (1.16%).", SEARCH)["grounded"]
    got = issues("The clearest scene is S2B_43PGQ_20260323_0_L2A (2.64%).", SEARCH)
    assert len(got) == 1 and "S2C_43PHQ_20260328_0_L2A has the lowest cloud_cover (1.16)" in got[0], got
    assert ok("The cloudiest is S2C_43PGQ_20260328_0_L2A (4.66%).", SEARCH)["grounded"]


def test_ties_pass():
    tied = str({"items": [{"id": "A_1_X", "cloud_cover": 0.02}, {"id": "B_1_X", "cloud_cover": 0.02}, {"id": "C_1_X", "cloud_cover": 2.5}]})
    assert ok("The clearest are A_1_X and B_1_X (0.02%).", tied)["grounded"]
    assert ok("The clearest is B_1_X (0.02%).", tied)["grounded"]


# ── weather days ─────────────────────────────────────────────────────────────

def test_weather_table_rows_pass_and_a_swapped_cell_is_flagged():
    good = "| 2026-03-09 | 30.1 | 19.4 | 0.0 | 11.7 |\n| 2026-03-10 | 29.9 | 20.3 | 1.5 | 8.7 |"
    assert ok(good, WEATHER)["grounded"]
    bad = "| 2026-03-09 | 30.1 | 19.4 | 0.0 | 8.7 |"   # 8.7 km/h is 2026-03-10's wind
    got = issues(bad, WEATHER)
    assert got and "belongs to 2026-03-10" in got[0], got


def test_month_day_superlatives_for_weather():
    assert ok("Hottest day: Mar 11, max 34.8 °C.", WEATHER)["grounded"]
    got = issues("Hottest day: Mar 12, max 31.2 °C.", WEATHER)
    assert got and "2026-03-11 has the highest temperature_2m_max (34.8)" in got[0], got
    assert ok("Coldest day: Mar 10, max only 29.9 °C.", WEATHER)["grounded"]       # lowest daily max
    assert ok("The wettest day was Mar 10 with 1.5 mm.", WEATHER)["grounded"]
    assert not ok("The wettest day was Mar 12 with 0.8 mm.", WEATHER)["grounded"]


# ── TerraMind ────────────────────────────────────────────────────────────────

def test_ranking_similarity_pairing_and_superlative():
    assert ok("The most similar scene is S2A_43QHV_20240128_0_L2A (cosine similarity 0.9991).", RANKING)["grounded"]
    assert issues("The most similar scene is S2A_43QHV_20240118_1_L2A (cosine similarity 0.9982).", RANKING)
    assert issues("emb_S2A_43QHV_20240118_1_L2A has cosine similarity 0.9991.", RANKING)   # the other scene's value


# ── things that must NOT be flagged ──────────────────────────────────────────

def test_comparisons_ranges_hedges_and_multi_scene_lines_are_left_alone():
    assert ok("S2C_43PHQ_20260328_0_L2A (1.16%) is clearer than the 4.66% scene.", SEARCH)["grounded"]
    assert ok("S2B_43PGQ_20260323_0_L2A (2.64%) is among the clearest.", SEARCH)["grounded"]
    assert ok("S2B_43PGQ_20260323_0_L2A is the second clearest at 2.64%.", SEARCH)["grounded"]
    assert ok("Winds were calm on Mar 10-12 (8.7–19.4 km/h).", WEATHER)["grounded"]
    assert ok("S2C_43PGQ_20260328_0_L2A (4.66%) and S2C_43PHQ_20260328_0_L2A (1.16%) share a tile pair.", SEARCH)["grounded"]


def test_numbers_that_match_no_record_are_left_to_the_existence_check():
    # 7.5 is nowhere in the data: the old check flags it as "number", pairing adds nothing
    got = ok("S2C_43PGQ_20260328_0_L2A has 7.5% cloud.", SEARCH)["unsupported"]
    assert got == ["number 7.5"], got


def test_no_records_no_pairing():
    geo = str({"best_match": {"display_name": "Paris, France", "lat": 48.8588897, "lon": 2.32}})
    assert ok("Paris is at 48.86, 2.32.", geo)["grounded"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
