"""Context-policy demo: a 7-turn session where the rolling summary fires, then a follow-up
that only the summary (or a fresh tool call) can answer.

With the defaults (summarise when the prompt passes 4,000 tokens, 2 verbatim turns kept) the
summary is written at the start of turn 7, on the same request that asks about turn 1.
Writes demo/demo_transcript_context.md and demo/demo_trace_context.jsonl.

Prereqs: MCP server + API running (README). Usage: python scripts/run_demo_context.py [--base URL]
"""

import argparse
import json
import time
from pathlib import Path

import httpx

DEMO = Path(__file__).resolve().parent.parent / "demo"
TURNS = [
    "What Sentinel-2 imagery is available for Hyderabad in January 2024 with less than 10% cloud cover?",
    "What was the weather like there during the same period?",
    "Where is Nairobi? Give me its coordinates.",
    "What was the weather in Nairobi from 2023-07-01 to 2023-07-07?",
    "What imagery collections can you search?",
    "Find Sentinel-2 scenes over Nairobi in March 2024 under 30% cloud, limit 3.",
    "Going back to the very first thing I asked: which city was it, what cloud-cover limit did I set, and what date range?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    sid = f"ctx-demo-{int(time.time())}"
    md = [f"# Context-policy demo (session `{sid}`)", ""]
    rows = []
    with httpx.Client(timeout=180) as c:
        for i, msg in enumerate(TURNS, 1):
            data = c.post(f"{args.base}/chat", json={"session_id": sid, "message": msg}).json()
            steps = [s for s in data["trace"] if s["step"] in ("tool_call", "context_summary", "verify")]
            rows += data["trace"]
            md += [f"## Turn {i}", f"**User:** {msg}", "", f"**Reply:** {data['reply']}", "", "Trace:"]
            for s in steps:
                extra = s["final_answer"] if s["step"] == "context_summary" else json.dumps(s.get("args"))
                md.append(f"- `{s['step']}` {s.get('tool_name') or ''} {extra}")
            md.append("")
            print(f"turn {i}: {[s['step'] + ':' + (s['tool_name'] or '') for s in steps]}")
    (DEMO / "demo_transcript_context.md").write_text("\n".join(md))
    (DEMO / "demo_trace_context.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
