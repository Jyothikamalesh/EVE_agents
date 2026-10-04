"""Offline test for the verifier detection benchmark. Run: python -m evals.test_detection

It pins what the benchmark currently measures, including the known weaknesses, so that a change to the
checker that improves or breaks detection shows up here and forces the documentation to be updated
(evals/SCENARIOS.md). A "miss" type that starts being caught is good news, but the prediction must be updated
deliberately, not silently.
"""

from collections import Counter

from evals.detection_fixtures import ERROR_TYPES, SCENARIOS, build
from evals.eval_detection import prf, run, summarize

# error types whose measured recall differs from the prediction written before the first run
KNOWN_UNEXPECTED = {
    # predicted "detect", measured 80%: a wrong average passes when it lands within one last-digit unit of some
    # number in the payload (15.9 is accepted because the data holds 15.8 and 16.0)
    "wrong_average_off_data": (0.6, 0.9),
}

RESULTS = run()
SUMMARY = summarize(RESULTS)


def test_fixtures_are_balanced_enough_to_mean_something():
    items = build()
    assert len(items) >= 250 and {i.label for i in items} == {"supported", "hallucinated"}
    per_type = Counter(i.error_type for i in items if i.label == "hallucinated")
    assert set(per_type) == set(ERROR_TYPES) and min(per_type.values()) >= 3
    for sid in SCENARIOS:
        rows = [i for i in items if i.scenario == sid]
        assert any(i.label == "supported" for i in rows) and any(i.label == "hallucinated" for i in rows), sid


def test_no_correct_reply_is_flagged():
    fa = SUMMARY["false_alarm_rate"]
    assert fa["flagged"] == 0, fa["items"]       # every correct reply (incl. advice and capability text) passes untouched


def test_error_types_it_is_meant_to_catch_are_caught():
    for et, t in SUMMARY["error_types"].items():
        if t["expected"] == "detect" and et not in KNOWN_UNEXPECTED:
            assert t["recall"] >= 0.9, (et, t["recall"])
    d = SUMMARY["predicted_detectable"]
    assert d["precision"] == 1.0 and d["recall"] >= 0.95


def test_known_weaknesses_are_still_weaknesses():
    for et, t in SUMMARY["error_types"].items():
        if t["expected"] == "miss":
            assert t["recall"] <= 0.34, f"{et} is now caught ({t['recall']:.0%}): update its prediction in detection_fixtures.py and SCENARIOS.md"
    assert SUMMARY["predicted_weaknesses"]["caught"] <= 3 and SUMMARY["predicted_weaknesses"]["positives"] >= 25


def test_the_one_prediction_that_was_wrong_is_documented():
    for et, (lo, hi) in KNOWN_UNEXPECTED.items():
        assert lo <= SUMMARY["error_types"][et]["recall"] <= hi, et


def test_random_wrong_numbers_pass_often_in_a_dense_payload():
    sw = SUMMARY["coincidence_sweep"]
    assert 0.2 <= sw["one_decimal_dense_payload"]["rate"] <= 0.5       # measured 33%
    assert sw["two_decimals_sparse_payload"]["rate"] <= 0.05           # measured 1%


def test_metric_arithmetic():
    m = prf(8, 2, 2)
    assert (m["tp"], m["fp"], m["fn"]) == (8, 2, 2) and all(abs(m[k] - 0.8) < 1e-9 for k in ("precision", "recall", "f1"))
    m = prf(0, 0, 0)
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    assert prf(3, 0, 0)["f1"] == 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
