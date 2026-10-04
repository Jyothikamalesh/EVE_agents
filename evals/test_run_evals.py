"""Offline tests for the eval runner's repeat statistics (no API, no model).

Run: python -m evals.test_run_evals
"""

import tempfile
from pathlib import Path

from evals import run_evals
from evals.run_evals import summarize, wilson_interval, write_reports


def run(cid, passed, cat="tool_use", fails=(), tools=("get_weather",), rep=1, verify=None):
    return {"id": cid, "category": cat, "passed": passed, "rep": rep, "session_id": "s",
            "turns": [{"turn": 1, "failures": list(fails), "tools": list(tools), "grounding": {"grounded": True},
                       "passed": passed, "verify": verify}]}


def repeated():
    out = []
    out += [run("steady", True, rep=i) for i in (1, 2, 3)]
    out += [run("flaky", True, rep=1), run("flaky", False, rep=2, fails=["expected tool x not called"]),
            run("flaky", False, rep=3, fails=["expected tool x not called"])]
    out += [run("broken", False, "context", ["reply matched forbidden /y/"], rep=i) for i in (1, 2, 3)]
    out += [{"id": "tm", "category": "a2a", "passed": True, "skipped": "tool(s) not bound: embed_scene", "turns": []}]
    return out


def test_all_runs_passing_is_not_100_percent():
    # the point of the interval: 5 of 5 only supports "above ~57%", 30 of 30 "above ~89%"
    assert round(wilson_interval(5, 5)[0], 3) == 0.566
    assert round(wilson_interval(10, 10)[0], 3) == 0.722
    assert round(wilson_interval(30, 30)[0], 3) == 0.886
    assert wilson_interval(5, 5)[1] == 1.0


def test_interval_edges():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0 and round(hi, 3) == 0.434
    lo, hi = wilson_interval(3, 6)
    assert abs((lo + hi) / 2 - 0.5) < 1e-9 and 0 < lo < 0.5 < hi < 1   # symmetric around an even split


def test_summary_counts_flaky_failing_and_skipped():
    s = summarize(repeated())
    assert s["per_case"] == {"steady": "3/3", "flaky": "1/3", "broken": "0/3"}
    assert s["flaky"] == ["flaky"] and s["failing"] == ["broken"] and s["skipped"] == ["tm"]
    assert s["cases"] == 3 and s["passed"] == 1                      # cases that passed every run
    assert s["runs"] == 9 and s["runs_passed"] == 4 and s["pass_rate"] == round(4 / 9, 3)
    assert s["case_stats"]["flaky"]["failures"] == {"expected tool x not called": 2}   # most common failures, counted
    assert s["by_category"] == {"tool_use": "4/6", "context": "0/3"}  # per run, skipped cases excluded


def test_single_run_keeps_the_old_meaning():
    s = summarize([run("a", True), run("b", False, fails=["boom"]), run("c", True, "context")])
    assert (s["cases"], s["passed"], s["runs"]) == (3, 2, 3)
    assert s["flaky"] == [] and s["failing"] == ["b"] and s["by_category"] == {"tool_use": "1/2", "context": "1/1"}


def test_report_shows_pass_counts_and_failure_counts():
    old = run_evals.RESULTS_DIR
    with tempfile.TemporaryDirectory() as d:
        run_evals.RESULTS_DIR = Path(d)
        try:
            res = repeated()
            path = write_reports(res, summarize(res), "t", "model-x", repeat=3)
            md = path.read_text()
        finally:
            run_evals.RESULTS_DIR = old
    assert "| flaky | 1/3 |" in md and "| steady | 3/3 |" in md and "| tm | SKIP |" in md
    assert "expected tool x not called (x2)" in md and "flaky: flaky" in md and "failing: broken" in md
    assert "95% CI" in md


def test_verifier_interventions_are_reported_separately_from_passes():
    clean = {"issues": [], "rewritten": False, "caveat": False}
    fixed = {"issues": ["number 16"], "rewritten": True, "caveat": False}
    left = {"issues": ["date 2024-02-29"], "rewritten": True, "caveat": True}
    res = [run("a", True, verify=clean), run("a", True, verify=fixed), run("b", True, verify=fixed), run("b", True, verify=left),
           run("c", True)]                                           # no verify step recorded: not counted
    s = summarize(res)
    assert s["runs_passed"] == 5 and s["passed"] == 3                   # every final reply passed...
    assert s["verifier"] == {"turns": 4, "flagged": 3, "rewritten": 3, "caveats": 1,
                             "flagged_by_case": {"a": 1, "b": 2}}      # ...but 3 of 4 needed the verifier
    line = run_evals.verifier_line(s)
    assert "flagged 3/4 turns (75%)" in line and "b (2)" in line and "a (1)" in line
    assert run_evals.verifier_line(summarize([run("x", True)])) == ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
