"""Answer verifier node for the ReAct graph (opt-in via ``compile(verify=True)``).

Runs once on the agent's final answer, never inside the tool loop:

1. ``check_groundedness`` — every date, scene ID and number in the reply must trace to a
   tool result, the rolling summary or the user's own words — plus a check that any tool
   the reply *cites* by name was actually called.
2. On failure, one rewrite pass (no tools bound): the model is shown this turn's tool
   results, its draft and the offending values, and must restate using only those.
3. If the rewrite is still ungrounded, the offending values are listed in a visible
   caveat appended to the reply instead of being passed off as verified.

The outcome is stored in ``state["verification"]`` so the service can trace it.
"""

from __future__ import annotations

import logging
import re
import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .context import _strip_thinking, turn_starts
from .grounding import check_groundedness, find_uncalled_tool_citations

logger = logging.getLogger(__name__)

TOOL_RESULT_CAP = 6000

REWRITE_SYSTEM = """You correct a draft answer so that every factual value in it is supported by the tool results.

Rules:
- Use ONLY values (dates, scene IDs, numbers, coordinates) that appear in the tool results below or in the user's question.
- Remove or rephrase any statement that relies on a listed unsupported value. If the data to answer is not in the tool results, say so plainly instead of answering from memory.
- Keep what you can still say truthfully: explanations of what you cannot do and why, limits of your tools, and suggestions or offers to the user. Only claims about results need support, so do not delete the rest.
- When you state a value from a tool, name the tool in parentheses, e.g. "cloud cover 3% (search_stac_items)".
- Keep the answer concise. Output only the corrected answer."""


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _issues(reply, outputs, users, tool_names, called, no_tools_this_turn, tool_texts=()) -> List[str]:
    res = check_groundedness(reply, outputs, users, mode="strict", capability_texts=tool_texts)
    bad = list(res["unsupported"])
    if no_tools_this_turn:
        # a no-tool turn legitimately says "3 tools", "step 2"; small bare integers aren't claims
        bad = [b for b in bad if not (b.startswith("number ") and re.fullmatch(r"-?\d{1,2}", b[7:]) and int(b[7:]) <= 10)]
    return bad + find_uncalled_tool_citations(reply, tool_names, called)


def make_verify_node(llm, tool_names: List[str], tool_texts: Optional[List[str]] = None):
    async def verify_fn(state) -> Dict[str, Any]:
        msgs = state["messages"]
        final = msgs[-1]
        if not isinstance(final, AIMessage) or getattr(final, "tool_calls", None):
            return {}
        starts = turn_starts(msgs)
        turn = msgs[starts[-1]:] if starts else msgs
        turn_tools = [m for m in turn if isinstance(m, ToolMessage)]
        outputs = [_text(m.content) for m in msgs if isinstance(m, ToolMessage)]
        if state.get("summary"):
            outputs.append(state["summary"])
        users = [_text(m.content) for m in msgs if isinstance(m, HumanMessage)]
        called = [m.name for m in msgs if isinstance(m, ToolMessage) and m.name]

        reply = _strip_thinking(_text(final.content))
        issues = _issues(reply, outputs, users, tool_names, called, not turn_tools, tool_texts or ())
        info: Dict[str, Any] = {"checked_chars": len(reply), "issues": issues, "rewritten": False, "caveat": False}
        if not issues:
            return {"verification": info}

        logger.warning("Verifier: %d unsupported value(s) in reply: %s", len(issues), issues)
        evidence = "\n".join(
            f"[{m.name}] {_text(m.content)[:TOOL_RESULT_CAP]}" for m in turn_tools
        ) or "(no tools were called this turn)"
        prompt = (
            f"User question:\n{users[-1] if users else ''}\n\nTool results this turn:\n{evidence}\n\n"
            f"Draft answer:\n{reply}\n\nUnsupported values: {issues}\n\nCorrected answer:"
        )
        try:
            resp = await llm.ainvoke([SystemMessage(content=REWRITE_SYSTEM), HumanMessage(content=prompt)])
            rewritten = _strip_thinking(_text(resp.content))
        except Exception:  # noqa: BLE001 - fall through to the caveat
            logger.exception("Verifier rewrite failed")
            rewritten = ""

        if rewritten:
            info["rewritten"] = True
            reply = rewritten
            issues = _issues(reply, outputs, users, tool_names, called, not turn_tools, tool_texts or ())
        info["remaining"] = issues
        if issues:
            info["caveat"] = True
            reply += (
                "\n\n⚠️ Could not verify against tool results: " + "; ".join(issues)
                + ". Treat these as unconfirmed."
            )
        return {"messages": [AIMessage(content=reply, id=final.id)], "verification": info}

    return verify_fn


def tool_capability_text(tool) -> str:
    """A tool's description plus its parameter schema: what the tool says about itself (limits, defaults)."""
    parts = [getattr(tool, "description", "") or ""]
    try:
        schema = tool.args_schema
        parts.append(json.dumps(schema if isinstance(schema, dict) else schema.model_json_schema()))
    except Exception:  # noqa: BLE001 - a tool without an introspectable schema just adds nothing
        pass
    return " ".join(parts)
