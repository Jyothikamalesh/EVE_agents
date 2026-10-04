"""Node-level trace of one graph run: which node ran, in what order, what each model call saw
and said, and what each tool call did.

A LangChain callback handler (passed in the run config, so the graph code is untouched) builds
a list of *node runs*:

    {"seq", "node", "duration_ms", "status", "error", "input", "output",
     "llm_calls": [{"prompt_summary", "prompt", "response", "tokens", "duration_ms"}],
     "tool_calls": [{"name", "args", "result", "error", "duration_ms", "remote"}]}

``input`` / ``output`` are small summaries for hover text; ``llm_calls[].prompt`` is the full
prompt (capped), shown on demand. Records are appended per turn to
``logs/node_runs/<session_id>.jsonl`` so they can be served again for reloaded sessions and
included in session exports.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage

from agents.graphs.context import COMPACT_NOTE
from agents.graphs.utils import tiktoken_counter

NODE_RUNS_DIR = Path(__file__).resolve().parent.parent / "logs" / "node_runs"
MAX_FIELD_CHARS = 20_000  # per message / tool result
MAX_PROMPT_CHARS = 100_000  # per model call
SUMMARY_MARK = "## Summary of earlier turns"


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content if isinstance(content, str) else str(content)


def _cap(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"… [truncated {len(text) - n} chars]"


def serialize_prompt(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Messages as sent to the model, each capped; the whole prompt is kept under MAX_PROMPT_CHARS."""
    role = {"SystemMessage": "system", "HumanMessage": "user", "AIMessage": "assistant", "ToolMessage": "tool"}
    out = []
    for m in messages:
        d: Dict[str, Any] = {"role": role.get(type(m).__name__, type(m).__name__), "content": _cap(_text(m.content), MAX_FIELD_CHARS)}
        if getattr(m, "tool_calls", None):
            d["tool_calls"] = [{"name": tc["name"], "args": tc["args"]} for tc in m.tool_calls]
        if getattr(m, "name", None) and d["role"] == "tool":
            d["name"] = m.name
        out.append(d)
    total = sum(len(d["content"]) for d in out)
    if total > MAX_PROMPT_CHARS:  # shrink the biggest messages first (old tool results)
        for d in sorted(out, key=lambda d: len(d["content"]), reverse=True):
            if total <= MAX_PROMPT_CHARS:
                break
            keep = max(500, len(d["content"]) // 4)
            total -= len(d["content"]) - keep
            d["content"] = _cap(d["content"], keep)
    return out


def summarize_prompt(messages: List[BaseMessage]) -> Dict[str, Any]:
    """What matters about a prompt for the context policy: size, summary block, compacted results."""
    sys_text = " ".join(_text(m.content) for m in messages if type(m).__name__ == "SystemMessage")
    try:
        tokens = tiktoken_counter(messages)
    except Exception:  # noqa: BLE001
        tokens = None
    def is_result(m: BaseMessage) -> bool:  # the graph may flatten ToolMessages into "[TOOL_RESULTS]" text
        return type(m).__name__ == "ToolMessage" or _text(m.content).startswith("[TOOL_RESULTS]")

    return {
        "messages": len(messages),
        "approx_tokens": tokens,
        "turns_in_prompt": sum(type(m).__name__ == "HumanMessage" and not is_result(m) for m in messages),
        "has_summary_block": SUMMARY_MARK in sys_text,
        "compacted_tool_results": sum(_text(m.content).count(COMPACT_NOTE) for m in messages if is_result(m)),
        "tool_results": sum(is_result(m) for m in messages),
    }


def _summarize_output(outputs: Any) -> Dict[str, Any]:
    if not isinstance(outputs, dict):
        return {}
    out: Dict[str, Any] = {}
    msgs = outputs.get("messages")
    if msgs:
        msgs = msgs if isinstance(msgs, list) else [msgs]
        calls = [tc["name"] for m in msgs for tc in (getattr(m, "tool_calls", None) or [])]
        if calls:
            out["requested_tools"] = calls
        last = next((m for m in reversed(msgs) if type(m).__name__ == "AIMessage" and not getattr(m, "tool_calls", None)), None)
        if last is not None:
            out["final_answer"] = _cap(_text(last.content), 400)
        results = [m for m in msgs if type(m).__name__ == "ToolMessage"]
        if results:
            out["tool_results"] = [{"name": m.name, "chars": len(_text(m.content))} for m in results]
    if outputs.get("summary"):
        out["summary_updated"] = True
        out["summarized_turns"] = outputs.get("summarized_turns")
    if outputs.get("verification") is not None:
        v = outputs["verification"]
        out["verification"] = {k: v.get(k) for k in ("issues", "rewritten", "caveat")}
    return out


class NodeTraceHandler(AsyncCallbackHandler):
    """Collects node runs for one graph invocation. ``remote_tools`` maps tool name -> remote agent label."""

    raise_error = False

    def __init__(self, remote_tools: Optional[Dict[str, str]] = None) -> None:
        self.remote_tools = remote_tools or {}
        self.runs: List[Dict[str, Any]] = []
        self._by_run_id: Dict[Any, Dict[str, Any]] = {}  # node-level chain runs
        self._parent: Dict[Any, Any] = {}  # run_id -> parent_run_id (every run)
        self._children: Dict[Any, Dict[str, Any]] = {}  # llm/tool run_id -> its record
        self._t0: Dict[Any, float] = {}

    # -- helpers
    def _node_of(self, run_id: Any) -> Optional[Dict[str, Any]]:
        seen = 0
        cur = self._parent.get(run_id)
        while cur is not None and seen < 50:
            if cur in self._by_run_id:
                return self._by_run_id[cur]
            cur = self._parent.get(cur)
            seen += 1
        return None

    # -- node (graph step) level
    async def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        self._parent[run_id] = parent_run_id
        node = (metadata or {}).get("langgraph_node")
        name = kwargs.get("name") or (serialized or {}).get("name")
        if node and name == node and not node.startswith("__"):
            run = {"seq": len(self.runs) + 1, "node": node, "started_at": time.time(), "duration_ms": None,
                   "status": "running", "error": None, "input": {}, "output": {}, "llm_calls": [], "tool_calls": []}
            self.runs.append(run)
            self._by_run_id[run_id] = run
            self._t0[run_id] = time.perf_counter()

    async def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        run = self._by_run_id.get(run_id)
        if run:
            run["duration_ms"] = round((time.perf_counter() - self._t0[run_id]) * 1000, 1)
            run["status"] = "ok"
            run["output"] = _summarize_output(outputs)

    async def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        run = self._by_run_id.get(run_id)
        if run:
            run["duration_ms"] = round((time.perf_counter() - self._t0[run_id]) * 1000, 1)
            run["status"] = "error"
            run["error"] = f"{type(error).__name__}: {error}"

    # -- model calls
    async def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        self._parent[run_id] = parent_run_id
        node = self._node_of(run_id)
        flat = messages[0] if messages else []
        call = {"prompt_summary": summarize_prompt(flat), "prompt": serialize_prompt(flat), "response": None,
                "tokens": None, "duration_ms": None, "error": None}
        self._children[run_id] = call
        self._t0[run_id] = time.perf_counter()
        if node is not None:
            node["llm_calls"].append(call)
            if not node["input"]:
                node["input"] = call["prompt_summary"]

    async def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        call = self._children.get(run_id)
        if not call:
            return
        call["duration_ms"] = round((time.perf_counter() - self._t0[run_id]) * 1000, 1)
        gen = response.generations[0][0] if response.generations and response.generations[0] else None
        msg = getattr(gen, "message", None)
        if msg is not None:
            call["response"] = {
                "content": _cap(_text(msg.content), 4000),
                "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in (getattr(msg, "tool_calls", None) or [])],
            }
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                call["tokens"] = {k: usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")}

    async def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        call = self._children.get(run_id)
        if call:
            call["duration_ms"] = round((time.perf_counter() - self._t0[run_id]) * 1000, 1)
            call["error"] = f"{type(error).__name__}: {error}"

    # -- tool calls
    async def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, inputs=None, **kwargs):
        self._parent[run_id] = parent_run_id
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        node = self._node_of(run_id) or next((r for r in reversed(self.runs) if r["node"] == "tools"), None)
        call = {"name": name, "args": inputs if inputs is not None else input_str, "result": None, "error": None,
                "duration_ms": None, "remote": self.remote_tools.get(name)}
        self._children[run_id] = call
        self._t0[run_id] = time.perf_counter()
        if node is not None:
            node["tool_calls"].append(call)

    async def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        call = self._children.get(run_id)
        if call:
            call["duration_ms"] = round((time.perf_counter() - self._t0[run_id]) * 1000, 1)
            content = output.content if isinstance(output, BaseMessage) else output
            call["result"] = _cap(content if isinstance(content, str) else json.dumps(content, default=str), MAX_FIELD_CHARS)

    async def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        call = self._children.get(run_id)
        if call:
            call["duration_ms"] = round((time.perf_counter() - self._t0[run_id]) * 1000, 1)
            call["error"] = f"{type(error).__name__}: {error}"


# ── persistence ──────────────────────────────────────────────────────────────


def _safe(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", session_id)[:150]


def save_turn(session_id: str, turn: int, runs: List[Dict[str, Any]]) -> None:
    NODE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with (NODE_RUNS_DIR / f"{_safe(session_id)}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"session_id": session_id, "turn": turn, "saved_at": time.time(), "node_runs": runs}, default=str) + "\n")


def load_turns(session_id: str) -> List[Dict[str, Any]]:
    """All recorded turns for a session, latest record winning per turn number, ordered by turn."""
    path = NODE_RUNS_DIR / f"{_safe(session_id)}.jsonl"
    if not path.exists():
        return []
    latest: Dict[int, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        latest[rec["turn"]] = rec
    return [latest[k] for k in sorted(latest)]
