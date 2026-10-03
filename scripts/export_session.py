"""Export one chat session as a single JSON file (messages with tool calls/results, API trace,
TerraMind agent log lines, embeddings it created), keyed by its session id.

    python scripts/export_session.py <session_id> [--out file.json] [--base http://127.0.0.1:8000]

Share the file: the UI's "Open a session file" replays it read-only. To follow a session in the
raw logs instead:  grep <session_id> logs/api.log logs/terramind.log; jq 'select(.session_id=="<id>")' logs/traces.jsonl logs/terramind.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--out")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    r = httpx.get(f"{args.base}/sessions/{args.session_id}/export", timeout=30)
    if r.status_code == 404:
        print(f"no session {args.session_id!r} on {args.base}", file=sys.stderr)
        return 1
    r.raise_for_status()
    out = Path(args.out or f"session-{args.session_id}.json")
    out.write_text(json.dumps(r.json(), indent=2))
    d = r.json()
    print(f"wrote {out}: {len(d['messages'])} messages, {len(d['trace'])} trace steps, "
          f"{len(d['terramind_log'])} TerraMind calls, {len(d['embeddings'])} embeddings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
