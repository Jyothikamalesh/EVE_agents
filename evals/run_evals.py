"""Run the eval set against a live API and score it.

Prereqs: the MCP server and API are running (see README "Run").

    python -m evals.run_evals                      # all cases
    python -m evals.run_evals --only tool_error_bad_date,imagery_basic
    python -m evals.run_evals --base http://127.0.0.1:18000 --db /path/to/checkpoints.sqlite

Per turn it checks tool use (from the API's inline trace), reply content
(regexes), and groundedness (reply values vs the tool outputs the model
actually saw, read back from the session's checkpoint). Writes
``evals/results/<timestamp>.json`` and ``.md``; exits 1 if any case fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from collections import defaultdict
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
    for p in turn.get("reply_none", []):
        if re.search(p, reply, re.I):
            fails.append(f"reply matched forbidden /{p}/")
    return fails


def run_case(case: dict, client: httpx.Client, base: str, saver, run_id: str) -> dict:
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
                grounding = check_groundedness(reply, outputs, user_texts, mode=mode, allow=turn.get("allow", ()))
                if not grounding["grounded"]:
                    fails.append(f"ungrounded claims: {grounding['unsupported']}")

        turns_out.append({
            "turn": i, "say": turn["say"], "reply": reply,
            "tools": [s["tool_name"] for s in trace if s.get("step") == "tool_call"],
            "tool_errors": [s["error"] for s in trace if s.get("step") == "tool_call" and s.get("error")],
            "grounding": grounding, "latency_s": round(time.perf_counter() - t0, 1),
            "passed": not fails, "failures": fails,
        })
    return {"id": case["id"], "category": case["category"], "session_id": sid,
            "passed": all(t["passed"] for t in turns_out), "turns": turns_out}


def summarize(results: list[dict]) -> dict:
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["passed"])
    graded = [t["grounding"] for r in results for t in r["turns"]
              if t.get("grounding") and "grounded" in t["grounding"]]
    return {
        "cases": len(results),
        "passed": sum(r["passed"] for r in results),
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in by_cat.items()},
        "groundedness": f"{sum(g['grounded'] for g in graded)}/{len(graded)} turns grounded",
    }


def write_reports(results: list[dict], summary: dict, stamp: str, model: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{stamp}.json").write_text(json.dumps({"summary": summary, "model": model, "results": results}, indent=2))
    lines = [f"# Eval run {stamp}", "", f"Model: `{model}`  ", f"**{summary['passed']}/{summary['cases']} cases passed** · {summary['groundedness']}", "",
             "| category | passed |", "|---|---|", *[f"| {c} | {v} |" for c, v in summary["by_category"].items()], "",
             "| case | result | tools (per turn) | notes |", "|---|---|---|---|"]
    for r in results:
        tools = " → ".join(",".join(t["tools"]) or "–" for t in r["turns"])
        notes = "; ".join(f for t in r["turns"] for f in t["failures"]) or ""
        lines.append(f"| {r['id']} | {'PASS' if r['passed'] else 'FAIL'} | {tools} | {notes} |")
    path = RESULTS_DIR / f"{stamp}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("EVAL_API_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--db", default=DEFAULT_DB, help="checkpoint SQLite the API writes to")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()

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
        for case in cases:
            res = run_case(case, client, args.base, saver, run_id)
            results.append(res)
            print(f"{'PASS' if res['passed'] else 'FAIL'}  {res['id']:<28} " +
                  " | ".join(",".join(t["tools"]) or "-" for t in res["turns"]))
            for t in res["turns"]:
                for f in t["failures"]:
                    print(f"        - turn {t['turn']}: {f}")

    summary = summarize(results)
    path = write_reports(results, summary, stamp, health.get("model", "?"))
    print(f"\n{summary['passed']}/{summary['cases']} passed · {summary['groundedness']}")
    print("by category:", summary["by_category"])
    print(f"report: {path}")
    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
