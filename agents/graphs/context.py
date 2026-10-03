"""Context-engineering helpers for the ReAct graph.

Three pieces, all off unless the caller opts in via ``ReactAgent.compile``:

* **Turns** — a turn is one ``HumanMessage`` plus everything up to the next one.
* **Tool-output compaction** — tool results from *earlier* turns are replaced by
  a compact form when the prompt is built. The stored history keeps the raw
  payload (the UI, evals and audits still see it); only what is sent to the
  model shrinks. The current turn is never compacted: the model is mid-loop.
* **Rolling summary** — once enough turns have aged out of the verbatim window,
  an LLM folds them into a running summary and they leave the prompt.

Compaction is per tool (a ``{tool_name: fn}`` registry supplied by the service,
so this library stays tool-agnostic) with a generic length cap as the fallback.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)

# parsed tool result (dict/list) -> smaller JSON-serialisable object
Compactor = Callable[[Any], Any]

DEFAULT_MAX_CHARS = 1200
COMPACT_NOTE = "[compacted; call the tool again for full detail]"

SUMMARY_SYSTEM = """You maintain a running summary of a conversation between a user and an Earth Observation assistant that has tools for geocoding, satellite-imagery (STAC) search and weather.

Update the summary by folding in the new turns.

Rules:
- Keep ONLY facts that appear in the conversation or in tool results. Never add, infer or guess.
- Keep: places (with lat/lon and bbox when given), date ranges, filters (e.g. max cloud cover), key results (counts; scene IDs and cloud cover of scenes the user may refer to again; notable weather values), tool errors and how they were resolved, the user's goals and preferences.
- Drop pleasantries and anything superseded by later turns.
- Plain bullet list, at most ~180 words. Output only the summary."""


# ── turns ────────────────────────────────────────────────────────────────────


def turn_starts(messages: List[BaseMessage]) -> List[int]:
    """Index of the first message of every turn (each ``HumanMessage``)."""
    return [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]


# ── compaction ───────────────────────────────────────────────────────────────


def _parse(text: str) -> Any:
    """Tool results are stored as ``str(result)``: JSON or a Python repr of a dict."""
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return None


def compact_content(
    tool_name: str,
    content: Any,
    compactors: Optional[Dict[str, Compactor]] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Compact one tool result: registered compactor first, else a length cap."""
    text = content if isinstance(content, str) else str(content)
    fn = (compactors or {}).get(tool_name)
    if fn is not None:
        parsed = _parse(text)
        if isinstance(parsed, (dict, list)):
            try:
                out = json.dumps(fn(parsed), ensure_ascii=False, separators=(",", ":"), default=str)
                return f"{out} {COMPACT_NOTE}"
            except Exception:  # noqa: BLE001 - never let compaction break a turn
                logger.exception("Compactor for %s failed; falling back to length cap", tool_name)
    if len(text) > max_chars:
        return f"{text[:max_chars]}… [truncated {len(text) - max_chars} chars; call the tool again for full detail]"
    return text


def compact_old_tool_messages(
    messages: List[BaseMessage],
    compactors: Optional[Dict[str, Compactor]] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[BaseMessage]:
    """Compact every ToolMessage that belongs to a turn before the current one."""
    starts = turn_starts(messages)
    boundary = starts[-1] if starts else 0
    out: List[BaseMessage] = []
    for i, m in enumerate(messages):
        if i < boundary and isinstance(m, ToolMessage):
            new = compact_content(m.name or "", m.content, compactors, max_chars)
            if new != m.content:
                m = ToolMessage(content=new, tool_call_id=m.tool_call_id, name=m.name, id=m.id)
        out.append(m)
    return out


# ── rolling summary ──────────────────────────────────────────────────────────


def render_turns(
    messages: List[BaseMessage],
    compactors: Optional[Dict[str, Compactor]] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Plain-text transcript of turns for the summarizer (tool results compacted)."""
    lines: List[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                lines.append(f"ASSISTANT called {tc['name']}({tc['args']})")
            if m.content and not getattr(m, "tool_calls", None):
                lines.append(f"ASSISTANT: {m.content}")
        elif isinstance(m, ToolMessage):
            lines.append(f"TOOL {m.name} result: {compact_content(m.name or '', m.content, compactors, max_chars)}")
    return "\n".join(lines)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


async def summarize_turns(
    llm,
    previous_summary: str,
    messages: List[BaseMessage],
    compactors: Optional[Dict[str, Compactor]] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Fold *messages* into *previous_summary*; returns the new summary text."""
    prompt = (
        f"Previous summary:\n{previous_summary or '(none)'}\n\n"
        f"New turns to fold in:\n{render_turns(messages, compactors, max_chars)}\n\n"
        "Updated summary:"
    )
    resp = await llm.ainvoke([SystemMessage(content=SUMMARY_SYSTEM), HumanMessage(content=prompt)])
    content = resp.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return _strip_thinking(str(content))


def summary_block(summary: str) -> Optional[str]:
    if not summary:
        return None
    return (
        "## Summary of earlier turns (no longer shown verbatim)\n"
        f"{summary}\n"
        "Treat this as context from earlier in the conversation. If you need exact detail "
        "(full scene lists, daily values), call the relevant tool again rather than guessing."
    )
