"""Run the eval set against a live API and score it.

Prereqs: the MCP server and API are running (see README "Run").

    python -m evals.run_evals                      # all cases
    python -m evals.run_evals --only tool_error_bad_date,imagery_basic
    python -m evals.run_evals --base http://127.0.0.1:18000 --db /path/to/checkpoints.sqlite

Per turn it checks tool use (from the API's inline trace), reply content
(regexes), and groundedness (reply values vs the tool outputs the model
actually saw, read back from the session's checkpoint). Writes
``evals/results/<timestamp>.json`` and ``.md``; exits 1 if any case fails.

The model is not deterministic even at temperature 0, so one pass per case says little about
reliability. ``--repeat N`` runs every case N times (each in a fresh session) and reports passes
per case (``4/5``), the overall pass rate with a 95% Wilson interval, and which cases are flaky
(passed sometimes) or failing (never passed). ``--min-pass-rate`` sets the exit-code threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import yaml
from langgraph.checkpoint.sqlite import SqliteSaver

from evals.groundedness import check_groundedness

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = Path(__file__).resolve().parent / "cases.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_DB = os.environ.get("CHECKPOINT_DB_PATH", str(ROOT / "data" / "checkpoints.sqlite"))


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def session_messages(saver: SqliteSaver | None, session_id: str) -> list:
    if saver is None:
        return []
    tup = saver.get_tuple({"configurable": {"thread_id": session_id}})
    return (tup.checkpoint.get("channel_values", {}).get("messages") or []) if tup else []


def post_chat(client: httpx.Client, base: str, session_id: str, message: str) -> dict:
    for attempt in range(4):
        resp = client.post(f"{base}/chat", json={"session_id": session_id, "message": message})
        if resp.status_code in (429, 500, 502, 503) and attempt < 3:
            time.sleep(2 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")


def check_turn(turn: dict, reply: str, trace: list) -> list[str]:
    """Return a list of failure strings (empty = pass)."""
    fails: list[str] = []
    calls = [s for s in trace if s.get("step") == "tool_call"]
    names = [c["tool_name"] for c in calls]

    for t in turn.get("tools", []):
        if t not in names:
            fails.append(f"expected tool {t} not called (called: {names or 'none'})")
    if turn.get("no_tools") and names:
        fails.append(f"expected no tool calls, got {names}")
    for t in turn.get("forbid_tools", []):
        if t in names:
            fails.append(f"forbidden tool {t} was called")
    for t, n in (turn.get("min_calls") or {}).items():
        if names.count(t) < n:
            fails.append(f"expected >= {n} calls to {t}, got {names.count(t)}")
    for spec in turn.get("args", []):
        hit = any(
            c["tool_name"] == spec["tool"] and re.search(spec["regex"], str((c.get("args") or {}).get(spec["key"], "")))
            for c in calls
        )
        if not hit:
            fails.append(f"no {spec['tool']} call with {spec['key']} ~ /{spec['regex']}/")
    if turn.get("tool_error") and not any(c.get("error") for c in calls):
        fails.append("expected a tool error in the trace, none recorded")
    if turn.get("cites_tool") and not any(re.search(rf"\b{re.escape(n)}\b", reply) for n in set(names)):
        fails.append("reply does not name a source tool for its values")
    any_pat = turn.get("reply_any")
    if any_pat and not any(re.search(p, reply, re.I) for p in any_pat):
        fails.append(f"reply matched none of {any_pat}")
    for p in turn.get("reply_all", []):
        if not re.search(p, reply, re.I):
            fails.append(f"reply did not match required /{p}/")
    verdict = next((s["args"] for s in trace if s.get("step") == "verify"), None)
    if turn.get("no_rewrite") and verdict and verdict.get("rewritten"):
        fails.append(f"the verifier rewrote the reply (issues: {verdict.get('issues')}); it should have passed untouched")
    for p in turn.get("reply_none", []):
        if re.search(p, reply, re.I):
            fails.append(f"reply matched forbidden /{p}/")
    return fails


def run_case(case: dict, client: httpx.Client, base: str, saver, run_id: str, capability_texts: list | tuple = ()) -> dict:
    sid = f"eval-{run_id}-{case['id']}-{uuid.uuid4().hex[:6]}"
    user_texts: list[str] = []
    turns_out = []
    for i, turn in enumerate(case["turns"], 1):
        user_texts.append(turn["say"])
        t0 = time.perf_counter()
        try:
            data = post_chat(client, base, sid, turn["say"])
        except Exception as exc:  # noqa: BLE001 - report, keep going
            turns_out.append({"turn": i, "say": turn["say"], "passed": False,
                              "failures": [f"request failed: {exc}"], "reply": "", "tools": []})
            break
        reply, trace = data["reply"], data["trace"]
        fails = check_turn(turn, reply, trace)

        grounding = None
        mode = turn.get("grounded", "strict")
        if mode:
            if saver is None:
                grounding = {"skipped": "checkpoint DB unreadable"}
            else:
                msgs = session_messages(saver, sid)
                outputs = [_text(m.content) for m in msgs if type(m).__name__ == "ToolMessage"]
                grounding = check_groundedness(reply, outputs, user_texts, mode=mode, allow=turn.get("allow", ()),
                                               capability_texts=capability_texts)
                if not grounding["grounded"]:
                    fails.append(f"ungrounded claims: {grounding['unsupported']}")

        turns_out.append({
            "turn": i, "say": turn["say"], "reply": reply,
            "tools": [s["tool_name"] for s in trace if s.get("step") == "tool_call"],
            "tool_errors": [s["error"] for s in trace if s.get("step") == "tool_call" and s.get("error")],
            "verify": next((s["args"] for s in trace if s.get("step") == "verify"), None),
            "grounding": grounding, "latency_s": round(time.perf_counter() - t0, 1),
            "passed": not fails, "failures": fails,
        })
    return {"id": case["id"], "category": case["category"], "session_id": sid,
            "passed": all(t["passed"] for t in turns_out), "turns": turns_out}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials; (0, 1) when there are no trials.

    k == n is not "100%": with n = 5 the lower bound is only about 57%, with n = 30 about 89%.
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(results: list[dict]) -> dict:
    """Aggregate runs (one entry per case per repetition; skipped cases carry ``skipped``)."""
    skipped = sorted({r["id"] for r in results if r.get("skipped")})
    ran = [r for r in results if not r.get("skipped")]
    per_case: dict[str, list[dict]] = defaultdict(list)
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in ran:
        per_case[r["id"]].append(r)
        by_cat[r["category"]].append(r["passed"])
    graded = [t["grounding"] for r in ran for t in r["turns"]
              if t.get("grounding") and "grounded" in t["grounding"]]
    case_stats = {}
    for cid, runs in per_case.items():
        fails = Counter(f for r in runs for t in r["turns"] for f in t["failures"])
        case_stats[cid] = {"category": runs[0]["category"], "runs": len(runs),
                           "passed": sum(r["passed"] for r in runs), "failures": dict(fails.most_common(3))}
    n_runs, k_runs = len(ran), sum(r["passed"] for r in ran)
    lo, hi = wilson_interval(k_runs, n_runs)
    # The replies scored above are the FINAL ones. Count how often the verifier had to step in, so a pass
    # that depended on a rewrite is not mistaken for a model that got it right the first time.
    vturns = [(r["id"], t["verify"]) for r in ran for t in r["turns"] if t.get("verify")]
    flagged_by_case: Counter = Counter(cid for cid, v in vturns if v.get("issues"))
    verifier = {"turns": len(vturns), "flagged": sum(flagged_by_case.values()),
                "rewritten": sum(bool(v.get("rewritten")) for _, v in vturns),
                "caveats": sum(bool(v.get("caveat")) for _, v in vturns),
                "flagged_by_case": dict(flagged_by_case)}
    return {
        "cases": len(case_stats),
        "skipped": skipped,
        "passed": sum(s["passed"] == s["runs"] for s in case_stats.values()),  # cases that passed every run
        "runs": n_runs,
        "runs_passed": k_runs,
        "pass_rate": round(k_runs / n_runs, 3) if n_runs else None,
        "ci95": [round(lo, 3), round(hi, 3)],
        "flaky": sorted(c for c, s in case_stats.items() if 0 < s["passed"] < s["runs"]),
        "failing": sorted(c for c, s in case_stats.items() if s["passed"] == 0),
        "per_case": {c: f"{s['passed']}/{s['runs']}" for c, s in case_stats.items()},
        "case_stats": case_stats,
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in by_cat.items()},
        "groundedness": f"{sum(g['grounded'] for g in graded)}/{len(graded)} turns grounded",
        "verifier": verifier,
    }


def verifier_line(summary: dict) -> str:
    v = summary["verifier"]
    if not v["turns"]:
        return ""
    where = ", ".join(f"{c} ({n})" for c, n in sorted(v["flagged_by_case"].items(), key=lambda kv: -kv[1])) or "none"
    return (f"verifier: flagged {v['flagged']}/{v['turns']} turns ({v['flagged'] / v['turns']:.0%}), rewrote {v['rewritten']}, "
            f"left {v['caveats']} caveats · flagged in: {where}")


def detection_line(d: dict | None) -> str:
    """One line for the verifier-as-detector benchmark (offline, no model): precision / recall / F1."""
    if not d:
        return ""
    o, e, fa = d["overall"], d["predicted_detectable"], d["false_alarm_rate"]
    f = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
    return (f"verifier as detector ({d['items']} labelled replies): precision {f(o['precision'])} · recall {f(o['recall'])} · F1 {f(o['f1'])}"
            f" (recall {f(e['recall'])} on the error types it is built for) · false alarms {fa['flagged']}/{fa['of']}")


def write_reports(results: list[dict], summary: dict, stamp: str, model: str, repeat: int = 1, detection: dict | None = None) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{stamp}.json").write_text(
        json.dumps({"summary": summary, "model": model, "repeat": repeat, "detection": detection, "results": results}, indent=2))
    skipped = f" · skipped: {', '.join(summary['skipped'])}" if summary["skipped"] else ""
    head = f"**{summary['passed']}/{summary['cases']} cases passed" + (" every run" if repeat > 1 else "") + f"** · {summary['groundedness']}{skipped}"
    lines = [f"# Eval run {stamp}", "", f"Model: `{model}`  ", head]
    if repeat > 1:
        lo, hi = summary["ci95"]
        lines += ["", f"{summary['runs_passed']}/{summary['runs']} runs passed ({summary['pass_rate']:.1%}, 95% CI {lo:.0%} to {hi:.0%}) "
                  f"· flaky: {', '.join(summary['flaky']) or 'none'} · failing: {', '.join(summary['failing']) or 'none'}"]
    vl = verifier_line(summary)
    if vl:
        lines += ["", vl + "  "]
        lines += ["(the pass rates above score the final replies, after any verifier rewrite)"]
    dl = detection_line(detection)
    if dl:
        lines += ["", dl + "  ", "(per scenario and error type: `python -m evals.eval_detection`; map in `evals/SCENARIOS.md`)"]
    lines += ["", "| category | passed |", "|---|---|", *[f"| {c} | {v} |" for c, v in summary["by_category"].items()], "",
              "| case | passes | tools (first run, per turn) | notes |", "|---|---|---|---|"]
    first = {}
    for r in results:
        first.setdefault(r["id"], r)
    for cid, r in first.items():
        if r.get("skipped"):
            lines.append(f"| {cid} | SKIP | | {r['skipped']} |")
            continue
        st = summary["case_stats"][cid]
        tools = " → ".join(",".join(t["tools"]) or "–" for t in r["turns"])
        notes = "; ".join(f"{m} (x{n})" if st["runs"] > 1 else m for m, n in st["failures"].items())
        lines.append(f"| {cid} | {st['passed']}/{st['runs']} | {tools} | {notes} |")
    path = RESULTS_DIR / f"{stamp}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("EVAL_API_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--db", default=DEFAULT_DB, help="checkpoint SQLite the API writes to")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--repeat", type=int, default=1, help="run every case N times (fresh session each) and report pass counts")
    ap.add_argument("--skip-detection", action="store_true", help="do not score the verifier on the labelled detection benchmark")
    ap.add_argument("--min-pass-rate", type=float, default=1.0,
                    help="exit 0 only if runs_passed/runs is at least this (default 1.0: every run passes)")
    args = ap.parse_args()
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2

    cases = yaml.safe_load(CASES_PATH.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
        if not cases:
            print(f"no cases match {wanted}", file=sys.stderr)
            return 2

    try:
        health = httpx.get(f"{args.base}/health", timeout=5).json()
    except httpx.HTTPError as exc:
        print(f"API not reachable at {args.base}: {exc}", file=sys.stderr)
        return 2

    saver = None
    if Path(args.db).exists():
        saver = SqliteSaver(sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, check_same_thread=False))
    else:
        print(f"WARNING: checkpoint DB {args.db} not found; groundedness will be skipped", file=sys.stderr)

    run_id = time.strftime("%H%M%S")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = []
    with httpx.Client(timeout=args.timeout) as client:
        bound = set(health.get("tools") or [])
        try:  # what the tools say about themselves: the same evidence the live verifier uses
            capability_texts = [x["text"] for x in client.get(f"{args.base}/tools").json()["tools"]]
        except Exception:  # noqa: BLE001 - older API without /tools: groundedness just gets less evidence
            capability_texts = []
        for case in cases:
            missing = [x for x in case.get("requires_tools", []) if x not in bound]
            if missing:  # e.g. the TerraMind cases when that agent isn't running: not a failure
                res = {"id": case["id"], "category": case["category"], "session_id": None, "passed": True,
                       "skipped": f"tool(s) not bound: {', '.join(missing)}", "turns": []}
                results.append(res)
                print(f"SKIP  {res['id']:<28} {res['skipped']}")
                continue
            for rep in range(1, args.repeat + 1):
                res = run_case(case, client, args.base, saver, run_id, capability_texts)
                res["rep"] = rep
                results.append(res)
                tag = f" [{rep}/{args.repeat}]" if args.repeat > 1 else ""
                print(f"{'PASS' if res['passed'] else 'FAIL'}  {res['id']:<28}{tag} " +
                      " | ".join(",".join(t["tools"]) or "-" for t in res["turns"]))
                for t in res["turns"]:
                    for f in t["failures"]:
                        print(f"        - turn {t['turn']}: {f}")

    summary = summarize(results)
    detection = None
    if not args.skip_detection:  # offline and cheap (no model): the verifier's precision / recall / F1 on labelled replies
        from evals.eval_detection import run as run_detection, summarize as summarize_detection
        detection = summarize_detection(run_detection())
    path = write_reports(results, summary, stamp, health.get("model", "?"), args.repeat, detection)
    note = f" · skipped {len(summary['skipped'])}: {', '.join(summary['skipped'])}" if summary["skipped"] else ""
    if args.repeat > 1:
        print("\npasses per case (lowest first):")
        for cid, frac in sorted(summary["per_case"].items(), key=lambda kv: int(kv[1].split("/")[0]) / int(kv[1].split("/")[1])):
            mark = "" if int(frac.split("/")[0]) == int(frac.split("/")[1]) else "   <-- not stable"
            print(f"  {cid:<34} {frac}{mark}")
        lo, hi = summary["ci95"]
        print(f"\n{summary['runs_passed']}/{summary['runs']} runs passed ({summary['pass_rate']:.1%}, 95% CI {lo:.0%} to {hi:.0%})"
              f" · {summary['passed']}/{summary['cases']} cases passed every run · {summary['groundedness']}{note}")
        print(f"flaky: {', '.join(summary['flaky']) or 'none'} · failing: {', '.join(summary['failing']) or 'none'}")
    else:
        print(f"\n{summary['passed']}/{summary['cases']} passed · {summary['groundedness']}{note}")
    vl = verifier_line(summary)
    if vl:
        print(vl)
    dl = detection_line(detection)
    if dl:
        print(dl)
    print("by category:", summary["by_category"])
    print(f"report: {path}")
    if not summary["runs"]:
        return 0
    return 0 if summary["runs_passed"] / summary["runs"] >= args.min_pass_rate - 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
