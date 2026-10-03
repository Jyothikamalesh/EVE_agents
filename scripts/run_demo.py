"""Reproduce the required demo: one session, three turns.

1. A question that needs a tool.
2. A follow-up that only works if context was kept ("there", "the same period").
3. A turn that forces a tool error (invalid date), to show the trace and the
   failure path without crashing the server.

Prereqs (see README "Run"):
    python -m mcp_server.server &
    uvicorn service.api:app --port 8000 &

Usage:
    source .venv/bin/activate
    python scripts/run_demo.py
"""

import json
import os
import sys
from pathlib import Path

import httpx

BASE = os.environ.get("DEMO_API_BASE", "http://127.0.0.1:8000")
SESSION_ID = "demo-session"
DEMO = Path(__file__).resolve().parent.parent / "demo"
LABELS = ["a question that needs a tool", "a follow-up that only works with context", "forced tool error"]

TURNS = [
    "What Sentinel-2 imagery is available for Hyderabad in January 2024 with less than 10% cloud cover?",
    "What was the weather like there during the same period?",
    "Call the weather tool directly with start_date='2024-02-30' and end_date='2024-02-30' "
    "for Hyderabad — I specifically want to see what the tool returns for that exact date, "
    "don't correct it yourself.",
]


def main() -> None:
    try:
        httpx.get(f"{BASE}/health", timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        print(f"API not reachable at {BASE}: {exc}", file=sys.stderr)
        print("Start it first: uvicorn service.api:app --port 8000", file=sys.stderr)
        sys.exit(1)

    md = ["# Demo transcript — single session, three required behaviours", "",
          f"Session id: `{SESSION_ID}`. Generated verbatim by `python scripts/run_demo.py`; raw trace: "
          "[`demo_trace.jsonl`](./demo_trace.jsonl).", ""]
    rows = []
    for i, message in enumerate(TURNS, 1):
        resp = httpx.post(
            f"{BASE}/chat",
            json={"session_id": SESSION_ID, "message": message},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        rows += data["trace"]
        md += [f"## Turn {i} — {LABELS[i - 1]}", f"**User:** {message}", "", f"**Agent:** {data['reply']}", "", "Trace:"]
        md += [f"- `{st['step']}` {st.get('tool_name') or ''} {json.dumps(st.get('args'))}"
               + (f" — ERROR: {st['error']}" if st.get("error") else "")
               for st in data["trace"] if st["step"] in ("tool_call", "verify", "context_summary")]
        md.append("")

        print(f"\n{'=' * 80}\nTurn {i}\n{'=' * 80}")
        print("USER:", message)
        print("\nREPLY:", data["reply"])
        print("\nTRACE:")
        for step in data["trace"]:
            print(" ", json.dumps({k: v for k, v in step.items() if k != "timestamp"}))

    health = httpx.get(f"{BASE}/health", timeout=5).json()
    print(f"\n{'=' * 80}\nServer health after the error turn:")
    print(health)
    md += [f"Server health after the error turn: `{json.dumps(health)}`", ""]
    (DEMO / "demo_transcript.md").write_text("\n".join(md))
    (DEMO / "demo_trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
