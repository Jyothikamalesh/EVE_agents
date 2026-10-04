"""Hallucination-detection benchmark for the verifier: precision, recall and F1 per scenario and error type.

    python -m evals.eval_detection              # print the tables and write evals/results/detection-<stamp>.{json,md}

No model and no network: every fixture in ``evals/detection_fixtures.py`` is a reply plus the tool outputs
it came from, labelled ``supported`` or ``hallucinated``. The decision under test is exactly the live
verifier's (``agents.graphs.verify._issues``): a reply is *flagged* if it returns any issue.

Positive class = hallucinated. For an error type, TP = corrupted replies flagged, FN = corrupted replies
missed, and FP/TN come from the correct replies of the same scenario (flagged / not flagged), so
precision answers "when it flags, is it right?" and the false-alarm rate answers "does it cry wolf?".
``--json`` prints the per-item results instead.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from agents.graphs.verify import _issues
from evals.detection_fixtures import ERROR_TYPES, EXPECT, SCENARIOS, TOOLS, build

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p is not None and r is not None and (p + r) > 0 else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def classify(item) -> list[str]:
    """The verifier's issues for this reply: exactly what the live node would compute."""
    return _issues(item.reply, list(item.outputs), list(item.users), TOOLS, list(item.called),
                   not item.called, list(item.tool_texts))


def run() -> list[dict]:
    out = []
    for it in build():
        issues = classify(it)
        out.append({"scenario": it.scenario, "error_type": it.error_type, "label": it.label, "flagged": bool(issues),
                    "issues": issues, "reply": it.reply, "note": it.note})
    return out


def summarize(results: list[dict]) -> dict:
    by_scn: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_scn[r["scenario"]].append(r)

    def conf(rows, positives):
        tp = sum(1 for r in positives if r["flagged"]); fn = len(positives) - tp
        neg = [r for r in rows if r["label"] == "supported"]
        fp = sum(1 for r in neg if r["flagged"])
        m = prf(tp, fp, fn)
        if not neg:               # no correct replies in this scope: precision is not measurable, do not report 100%
            m["precision"] = m["f1"] = None
        return {**m, "positives": len(positives), "negatives": len(neg), "tn": len(neg) - fp}

    scenarios = {}
    for sid, rows in sorted(by_scn.items(), key=lambda kv: int(kv[0][1:])):
        scenarios[sid] = conf(rows, [r for r in rows if r["label"] == "hallucinated"])
    types = {}
    for et, (sid, expected, desc) in ERROR_TYPES.items():
        rows = by_scn[sid]
        pos = [r for r in rows if r["error_type"] == et]
        c = conf(rows, pos)
        lo, hi = wilson(c["tp"], c["positives"])
        recall = c["recall"]
        types[et] = {**c, "scenario": sid, "expected": expected, "description": desc, "recall_ci95": [round(lo, 3), round(hi, 3)],
                     "as_predicted": (recall is not None) and ((recall >= 0.9) if expected == "detect" else (recall <= 0.34))}
    pos_all = [r for r in results if r["label"] == "hallucinated"]
    overall = conf(results, pos_all)
    detect_types = {et for et, v in EXPECT.items() if v == "detect"}
    detect_rows = [r for r in results if r["label"] == "hallucinated" and r["error_type"] in detect_types]
    miss_rows = [r for r in results if r["label"] == "hallucinated" and r["error_type"] not in detect_types]
    neg = [r for r in results if r["label"] == "supported"]
    false_alarms = [r for r in neg if r["flagged"]]
    return {
        "items": len(results), "supported": len(neg), "hallucinated": len(pos_all),
        "overall": overall,
        "predicted_detectable": {**prf(sum(r["flagged"] for r in detect_rows), len(false_alarms), sum(not r["flagged"] for r in detect_rows)),
                                 "positives": len(detect_rows)},
        "predicted_weaknesses": {"positives": len(miss_rows), "caught": sum(r["flagged"] for r in miss_rows)},
        "false_alarm_rate": {"flagged": len(false_alarms), "of": len(neg), "rate": len(false_alarms) / len(neg) if neg else None,
                             "items": [{"scenario": r["scenario"], "note": r["note"], "issues": r["issues"], "reply": r["reply"][:140]} for r in false_alarms]},
        "scenarios": scenarios, "error_types": types, "coincidence_sweep": sweep(),
    }


def sweep() -> dict:
    """How often does a RANDOM wrong number pass? A number is accepted if it is within one unit of its last
    digit of any number in the tool output (this forgives rounding vs truncation), so the answer depends on
    how many numbers the payload holds and how many decimals the reply uses."""
    from evals.detection_fixtures import SEARCH_OUT, WEATHER_OUT

    def rate(values, make, outputs, called, true):
        wrong = [v for v in values if v not in true]
        passed = sum(not _issues(make(v), [outputs], ["q"], TOOLS, [called], False, []) for v in wrong)
        return {"tried": len(wrong), "passed": passed, "rate": passed / len(wrong)}

    one = rate([round(x / 10, 1) for x in range(50, 281)], lambda v: f"The average max temperature was {v} °C (get_weather).",
               WEATHER_OUT, "get_weather", {15.4})
    two = rate([round(x / 100, 2) for x in range(0, 5001)], lambda v: f"Scene S2B_29SMC_20240114_0_L2A had {v:.2f}% cloud cover (search_stac_items).",
               SEARCH_OUT, "search_stac_items", {0.64})
    return {"one_decimal_dense_payload": one, "two_decimals_sparse_payload": two}


def fmt(x):
    return "n/a" if x is None else f"{x:.0%}"


def report_text(s: dict) -> str:
    L = []
    o = s["overall"]
    L.append(f"{s['items']} labelled replies: {s['hallucinated']} hallucinated, {s['supported']} supported "
             f"(synthetic Lisbon data, not the data the checker was developed on)\n")
    L.append("OVERALL (positive = hallucinated; flagged = verifier would intervene)")
    L.append(f"  precision {fmt(o['precision'])}   recall {fmt(o['recall'])}   F1 {fmt(o['f1'])}   "
             f"(TP {o['tp']}, FP {o['fp']}, FN {o['fn']}, TN {o['tn']})")
    d, w, fa = s["predicted_detectable"], s["predicted_weaknesses"], s["false_alarm_rate"]
    L.append(f"  on the error types it is meant to catch:  precision {fmt(d['precision'])}  recall {fmt(d['recall'])}  F1 {fmt(d['f1'])}  ({d['positives']} items)")
    L.append(f"  on the known weaknesses:                 caught {w['caught']} of {w['positives']}")
    hi_fa = wilson(fa["flagged"], fa["of"])[1]
    L.append(f"  false alarms on correct replies:         {fa['flagged']} of {fa['of']} ({fmt(fa['rate'])}; 95% upper bound {hi_fa:.0%}, the sample is small)\n")
    L.append("PER SCENARIO")
    L.append(f"  {'scenario':<5} {'title':<64} {'pos':>4} {'neg':>4} {'prec':>6} {'recall':>7} {'F1':>6}")
    for sid, c in s["scenarios"].items():
        L.append(f"  {sid:<5} {SCENARIOS[sid][0][:63]:<64} {c['positives']:>4} {c['negatives']:>4} {fmt(c['precision']):>6} {fmt(c['recall']):>7} {fmt(c['f1']):>6}")
    L.append("\nPER ERROR TYPE (recall with 95% interval; 'predicted' was written before the first run)")
    L.append(f"  {'error type':<28} {'scn':<4} {'n':>3} {'recall':>7} {'95% CI':>11} {'predicted':>9}  result")
    for et, t in s["error_types"].items():
        ci = f"{t['recall_ci95'][0]:.0%}-{t['recall_ci95'][1]:.0%}"
        L.append(f"  {et:<28} {t['scenario']:<4} {t['positives']:>3} {fmt(t['recall']):>7} {ci:>11} {t['expected']:>9}  {'as predicted' if t['as_predicted'] else 'UNEXPECTED'}")
    sw = s["coincidence_sweep"]
    L.append("\nHOW OFTEN A RANDOM WRONG NUMBER PASSES (numbers within one last-digit unit of any number in the tool output are accepted)")
    for k, label in (("one_decimal_dense_payload", "1 decimal, 10-day weather payload (about 40 numbers)"), ("two_decimals_sparse_payload", "2 decimals, 6-scene search payload")):
        v = sw[k]
        L.append(f"  {label:<58} {v['passed']:>4} of {v['tried']:>4} wrong values pass ({v['rate']:.0%})")
    if fa["items"]:
        L.append("\nFALSE ALARMS (correct replies the verifier would have rewritten)")
        for f in fa["items"]:
            L.append(f"  [{f['scenario']}] {f['note']}: {f['issues']}  <- {f['reply']!r}")
    return "\n".join(L)


def report_md(s: dict, stamp: str) -> str:
    o = s["overall"]
    L = [f"# Verifier detection benchmark ({stamp})", "",
         f"{s['items']} labelled replies ({s['hallucinated']} hallucinated, {s['supported']} supported); synthetic Lisbon data.", "",
         f"**Overall:** precision {fmt(o['precision'])}, recall {fmt(o['recall'])}, F1 {fmt(o['f1'])} "
         f"(TP {o['tp']}, FP {o['fp']}, FN {o['fn']}, TN {o['tn']}). False alarms on correct replies: "
         f"{s['false_alarm_rate']['flagged']} of {s['false_alarm_rate']['of']}. A random wrong number passes "
         f"{s['coincidence_sweep']['one_decimal_dense_payload']['rate']:.0%} of the time with 1 decimal in a dense payload and "
         f"{s['coincidence_sweep']['two_decimals_sparse_payload']['rate']:.1%} with 2 decimals in a sparse one.", "",
         "| scenario | title | hallucinated | correct | precision | recall | F1 |", "|---|---|---|---|---|---|---|"]
    for sid, c in s["scenarios"].items():
        L.append(f"| {sid} | {SCENARIOS[sid][0]} | {c['positives']} | {c['negatives']} | {fmt(c['precision'])} | {fmt(c['recall'])} | {fmt(c['f1'])} |")
    L += ["", "| error type | scenario | n | recall | 95% CI | predicted | result |", "|---|---|---|---|---|---|---|"]
    for et, t in s["error_types"].items():
        L.append(f"| {et} | {t['scenario']} | {t['positives']} | {fmt(t['recall'])} | {t['recall_ci95'][0]:.0%}-{t['recall_ci95'][1]:.0%} | {t['expected']} | {'as predicted' if t['as_predicted'] else '**UNEXPECTED**'} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print per-item results as JSON")
    ap.add_argument("--no-write", action="store_true", help="do not write the report files")
    args = ap.parse_args()
    results = run()
    if args.json:
        print(json.dumps(results, indent=1))
        return 0
    s = summarize(results)
    print(report_text(s))
    if not args.no_write:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (RESULTS_DIR / f"detection-{stamp}.json").write_text(json.dumps({"summary": s, "results": results}, indent=1))
        (RESULTS_DIR / f"detection-{stamp}.md").write_text(report_md(s, stamp))
        print(f"\nreport: {RESULTS_DIR / f'detection-{stamp}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
