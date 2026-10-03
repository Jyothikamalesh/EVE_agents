"""Per-request tool-call tracing (assignment section 2.4).

Each tool handed to the agent is wrapped so every invocation is timed and
logged with: session id, step name, tool name, arguments, duration, and
error (if any). Entries are appended to ``logs/traces.jsonl`` (one request
leaves a trace readable after the process exits) and also returned inline
in the API response for the demo (so a reply can be matched to its trace
without grepping a log file).

A failing tool must not crash the process: the wrapper logs the error and
re-raises, but ``agents.graphs.base.AgentGraph.make_tools_node`` is what
actually catches that and turns it into a ``ToolMessage`` the agent sees —
this module only owns observability, not recovery.
"""

import json
import logging
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)

TRACE_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "traces.jsonl"

# Holds the list the *current* request's trace steps get appended to.
# A contextvar (not a plain module global) so concurrent requests on
# different asyncio tasks don't interleave each other's trace entries.
_current_trace: ContextVar[List[Dict[str, Any]] | None] = ContextVar(
    "_current_trace", default=None
)


def start_trace() -> List[Dict[str, Any]]:
    """Begin a new trace for one request; returns the list steps will land in."""
    steps: List[Dict[str, Any]] = []
    _current_trace.set(steps)
    return steps


def _write_entry(entry: Dict[str, Any]) -> None:
    TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def record_step(
    *,
    session_id: str,
    step: str,
    tool_name: str | None = None,
    args: Dict[str, Any] | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
    final_answer: str | None = None,
) -> None:
    entry = {
        "timestamp": time.time(),
        "session_id": session_id,
        "step": step,
        "tool_name": tool_name,
        "args": args,
        "duration_ms": duration_ms,
        "error": error,
        "final_answer": final_answer,
    }
    steps = _current_trace.get()
    if steps is not None:
        steps.append(entry)
    _write_entry(entry)
    if error:
        logger.error(
            "Tool failed: session=%s tool=%s error=%s", session_id, tool_name, error
        )
    else:
        logger.info(
            "Trace step: session=%s step=%s tool=%s duration_ms=%s",
            session_id,
            step,
            tool_name,
            duration_ms,
        )


def _unwrap_mcp_result(result: Any) -> Any:
    """Collapse langchain-mcp-adapters' (content_blocks, artifact) tuple to plain text/JSON.

    ``MultiServerMCPClient`` tools return ``(content, artifact)`` for
    ``response_format="content_and_artifact"``, where ``content`` is a list
    of ``{"type": "text", "text": ...}`` blocks. Re-wrapping these tools (see
    below) loses that declared response_format, so left alone the raw tuple
    would land in the agent's ``ToolMessage`` as noisy repr text. Unwrap it
    to the single text block's contents (parsed as JSON when possible) so
    the model — and the trace log — sees the same clean structure the
    direct-import LangChain tools return.
    """
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], list)
        and result[0]
        and isinstance(result[0][0], dict)
        and result[0][0].get("type") == "text"
    ):
        text = result[0][0]["text"]
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
    return result


def wrap_tool_with_tracing(tool: BaseTool, session_id_getter) -> BaseTool:
    """Return a copy of *tool* whose async invocation is timed and logged.

    ``session_id_getter`` is a zero-arg callable returning the session id for
    the in-flight request (read from the same contextvar the API sets before
    calling the graph), since tool coroutines don't otherwise see it.
    """
    original_coroutine = tool.coroutine

    async def traced_coroutine(**kwargs):
        session_id = session_id_getter()
        start = time.perf_counter()
        try:
            result = _unwrap_mcp_result(await original_coroutine(**kwargs))
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            record_step(
                session_id=session_id,
                step="tool_call",
                tool_name=tool.name,
                args=kwargs,
                duration_ms=round(duration_ms, 1),
                error=str(exc),
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        record_step(
            session_id=session_id,
            step="tool_call",
            tool_name=tool.name,
            args=kwargs,
            duration_ms=round(duration_ms, 1),
        )
        return result

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=traced_coroutine,
    )
