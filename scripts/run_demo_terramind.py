"""TerraMind A2A demo: the EO agent delegates embedding work to the remote TerraMind agent.

Three turns in one session: search + embed three scenes, rank them against the first, then
locate where the first and the least similar one differ. Writes demo/demo_transcript_terramind.md
and demo/demo_trace_terramind.jsonl.

Prereqs: MCP server, `python -m terramind_agent.server` and the API are running (README).
Usage: python scripts/run_demo_terramind.py [--base URL]
"""

import argparse
import json
import time
from pathlib import Path

import httpx

DEMO = Path(__file__).resolve().parent.parent / "demo"
TURNS = [
    "Find three Sentinel-2 scenes over Hyderabad in January 2024 with under 5% cloud, all on the same MGRS tile, and embed each of them with TerraMind.",
    "Rank the other two by how similar they are to the first one.",
    "Where do the first scene and the least similar one differ most? Is the per-tile comparison valid for them?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    sid = f"terramind-demo-{int(time.time())}"
    md = ["# TerraMind A2A demo", "",
          f"Session `{sid}`. The EO agent's `embed_scene`, `compare_embeddings` and `rank_similar` tools are A2A calls to the "
          "separate TerraMind agent (`terramind_agent/`); the (196, 192) tensors stay in that agent, only embedding ids and "
          "summary numbers reach the LLM. Generated verbatim by `python scripts/run_demo_terramind.py`.", ""]
    rows = []
    with httpx.Client(timeout=300) as c:
        for i, msg in enumerate(TURNS, 1):
            data = c.post(f"{args.base}/chat", json={"session_id": sid, "message": msg}).json()
            rows += data["trace"]
            md += [f"## Turn {i}", f"**User:** {msg}", "", f"**Agent:** {data['reply']}", "", "Trace:"]
            for s in data["trace"]:
                if s["step"] in ("tool_call", "verify"):
                    md.append(f"- `{s['step']}` {s.get('tool_name') or ''} {json.dumps(s.get('args'))}"
                              + (f" ({s['duration_ms']} ms)" if s.get("duration_ms") else "")
                              + (f" — ERROR: {s['error']}" if s.get("error") else ""))
            md.append("")
            print(f"turn {i}: {[s['tool_name'] for s in data['trace'] if s['step'] == 'tool_call']}")
    (DEMO / "demo_transcript_terramind.md").write_text("\n".join(md))
    (DEMO / "demo_trace_terramind.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
