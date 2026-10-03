"""Session view and export: everything about one chat session, traceable by its session id.

``serialize_messages`` turns the checkpointed LangChain messages into plain JSON (what the UI
renders). ``export_session`` adds the API trace and the TerraMind agent's log lines for that
session and the metadata of the embeddings it created, so one JSON file shows the whole test.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .tracing import TRACE_LOG_PATH

ROOT = Path(__file__).resolve().parent.parent
TERRAMIND_LOG = Path(os.environ.get("TERRAMIND_LOG_PATH", str(ROOT / "logs" / "terramind.jsonl")))
EMBEDDINGS_DIR = Path(os.environ.get("TERRAMIND_STORE_DIR", str(ROOT / "data" / "embeddings")))


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        kind = type(m).__name__
        if kind == "HumanMessage":
            out.append({"role": "user", "content": _text(m.content)})
        elif kind == "AIMessage":
            out.append({
                "role": "assistant", "content": _text(m.content),
                "tool_calls": [{"id": tc.get("id"), "name": tc["name"], "args": tc["args"]} for tc in (m.tool_calls or [])],
            })
        elif kind == "ToolMessage":
            out.append({"role": "tool", "name": m.name, "tool_call_id": m.tool_call_id, "content": _text(m.content)})
    return out


def _jsonl_for(path: Path, session_id: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("session_id") == session_id:
                rows.append(row)
    return rows


def session_embeddings(session_id: str) -> List[Dict[str, Any]]:
    if not EMBEDDINGS_DIR.exists():
        return []
    metas = []
    for p in sorted(EMBEDDINGS_DIR.glob("*.json")):
        try:
            meta = json.loads(p.read_text())
        except ValueError:
            continue
        if meta.get("session_id") == session_id:
            metas.append(meta)
    return metas


def export_session(session_id: str, state_values: Dict[str, Any], model: str) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "messages": serialize_messages(state_values.get("messages") or []),
        "summary": state_values.get("summary") or None,
        "summarized_turns": state_values.get("summarized_turns") or 0,
        "trace": _jsonl_for(TRACE_LOG_PATH, session_id),
        "terramind_log": _jsonl_for(TERRAMIND_LOG, session_id),
        "embeddings": session_embeddings(session_id),
    }
