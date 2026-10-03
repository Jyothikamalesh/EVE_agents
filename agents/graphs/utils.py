"""Shared utilities for agent graphs — no backend dependencies.

Contains text-format tool-call parsing (Mistral/EVE-Instruct), message
reformatting, SSE label generation, token counting, and tool input schemas.

Only depends on standard library + langchain-core + pydantic.
Safe to import from standalone scripts/notebooks.
"""

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

_TEXT_TOOL_CALL_MARKER = "[TOOL_CALLS]"


# ─── SSE label generation ─────────────────────────────────────────────────────


def tool_call_label(tool_name: str) -> str:
    """Return a human-readable label for a streaming tool-call event."""
    if "knowledge_base" in tool_name:
        return "Searching knowledge base"
    if "wiley" in tool_name.lower():
        return "Searching Wiley Gateway"
    pretty = tool_name.replace("_", " ").replace("-", " ").strip()
    return f"Calling {pretty}" if pretty else "Calling tool"


# ─── Text-format tool-call parsing (Mistral / EVE-Instruct) ───────────────────


def _parse_json_array_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse ``[{"name": ..., "arguments": {...}}, ...]`` (seen from Qwen3 on Groq).

    Distinct from the bare Mistral format (``name{json}``, no enclosing list) —
    this is a straight JSON array of ``{"name", "arguments"}`` objects. Returns
    ``None`` (not ``[]``) when *text* isn't this format, so callers can fall
    back to the bare-format parser rather than treating "no calls found" as
    "zero tool calls intended".
    """
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    calls: List[Dict[str, Any]] = []
    for idx, entry in enumerate(parsed):
        if not isinstance(entry, dict) or "name" not in entry:
            return None
        calls.append(
            {
                "id": f"call_{idx}_{entry['name']}",
                "name": entry["name"],
                "args": entry.get("arguments", entry.get("args", {})),
                "type": "tool_call",
            }
        )
    return calls


_XML_NAME_RE = re.compile(r"<tool_name>\s*([A-Za-z_]\w*)\s*</tool_name>|<function=([A-Za-z_]\w*)>")
_XML_PARAM_RE = re.compile(r"<parameter=([A-Za-z_]\w*)>(.*?)</parameter>", re.S)
_XML_SPLIT_RE = re.compile(r"(?=<tool_use>)|(?=<tool_call>)|(?=<function=)")


def _parse_xml_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse XML-style calls some open models emit as plain text instead of native tool calls:

        <tool_use><tool_name>geocode_location</tool_name><parameter=query>Paris</parameter></function></tool_call>
        <tool_call><function=get_weather><parameter=lat>48.85</parameter>...</function></tool_call>

    Parameter values are JSON-decoded when they parse (numbers, lists), else kept as strings.
    """
    calls: List[Dict[str, Any]] = []
    for block in _XML_SPLIT_RE.split(content):
        m = _XML_NAME_RE.search(block)
        if not m:
            continue
        name = m.group(1) or m.group(2)
        args: Dict[str, Any] = {}
        for key, raw in _XML_PARAM_RE.findall(block):
            raw = raw.strip()
            try:
                args[key] = json.loads(raw)
            except ValueError:
                args[key] = raw
        calls.append({"id": f"call_{len(calls)}_{name}", "name": name, "args": args, "type": "tool_call"})
    return calls


def parse_text_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse text-format tool calls into structured dicts.

    Handles three formats seen in the wild. Two follow a ``[TOOL_CALLS]`` marker:

    - Mistral/EVE-Instruct bare format: ``tool_name{"key": "val"} ...``
    - JSON array format (e.g. Qwen3 on Groq): ``[{"name": ..., "arguments": {...}}]``

    and the XML style (``<tool_use>`` / ``<function=...>``), which has no marker.

    Returns a list compatible with ``AIMessage.tool_calls``.
    """
    if _TEXT_TOOL_CALL_MARKER not in content:
        return _parse_xml_tool_calls(content)

    text = content[
        content.index(_TEXT_TOOL_CALL_MARKER) + len(_TEXT_TOOL_CALL_MARKER) :
    ].strip()

    json_array_calls = _parse_json_array_tool_calls(text)
    if json_array_calls is not None:
        return json_array_calls

    calls: List[Dict[str, Any]] = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i] in " \t\n,":
            i += 1
        if i >= len(text):
            break
        m = re.match(r"([A-Za-z_]\w*)", text[i:])
        if not m:
            break
        name = m.group(1)
        i += m.end()
        while i < len(text) and text[i] in " \t":
            i += 1
        if i < len(text) and text[i] == "{":
            depth, j = 0, i
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            args = json.loads(text[i : j + 1])
                        except Exception:
                            args = {}
                        calls.append(
                            {
                                "id": f"call_{len(calls)}_{name}",
                                "name": name,
                                "args": args,
                                "type": "tool_call",
                            }
                        )
                        i = j + 1
                        break
                j += 1
            else:
                break
        else:
            calls.append(
                {
                    "id": f"call_{len(calls)}_{name}",
                    "name": name,
                    "args": {},
                    "type": "tool_call",
                }
            )
    return calls


def has_text_tool_call(content: str) -> bool:
    """Return True if *content* contains a text-format tool call marker."""
    return _TEXT_TOOL_CALL_MARKER in content


# ─── Message history sanitisation ─────────────────────────────────────────────


def reformat_messages_for_text_tool_model(
    messages: List[Any],
    *,
    AIMessage: Any = None,
    ToolMessage: Any = None,
    HumanMessage: Any = None,
) -> List[Any]:
    """Convert structured tool_calls/ToolMessages back to plain-text format.

    Pass LangChain message classes explicitly if calling from a context where
    they may not be installed, or leave as None to auto-import.
    """
    if AIMessage is None or ToolMessage is None or HumanMessage is None:
        from langchain_core.messages import (
            AIMessage as _AI,
            HumanMessage as _H,
            ToolMessage as _T,
        )

        AIMessage = AIMessage or _AI
        ToolMessage = ToolMessage or _T
        HumanMessage = HumanMessage or _H

    result: List[Any] = []
    for msg in messages:
        if (
            isinstance(msg, AIMessage)
            and not msg.content
            and getattr(msg, "tool_calls", None)
        ):
            calls = [
                {"name": tc["name"], "arguments": tc.get("args", {})}
                for tc in msg.tool_calls
            ]
            result.append(AIMessage(content=f"[TOOL_CALLS] {json.dumps(calls)}"))
        elif isinstance(msg, ToolMessage):
            result.append(HumanMessage(content=f"[TOOL_RESULTS]\n{msg.content}"))
        else:
            result.append(msg)
    return result


def strip_content_from_tool_call_messages(
    messages: List[Any],
    *,
    AIMessage: Any = None,
) -> List[Any]:
    """Strip text content from AIMessages that also carry tool_calls.

    Some APIs (Mistral) reject assistant messages with both non-empty
    content AND tool_calls.
    """
    if AIMessage is None:
        from langchain_core.messages import AIMessage as _AI

        AIMessage = _AI

    return [
        AIMessage(content="", tool_calls=m.tool_calls, id=m.id)
        if (isinstance(m, AIMessage) and m.content and getattr(m, "tool_calls", None))
        else m
        for m in messages
    ]


# ─── Token counting ───────────────────────────────────────────────────────────


def tiktoken_counter(messages: List[Any]) -> int:
    """Approximate token count using tiktoken cl100k_base."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, list):
                content = "".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            total += len(enc.encode(str(content))) + 4
        return total
    except Exception:
        return sum(len(str(getattr(m, "content", m))) // 4 for m in messages)


# ─── Tool input schemas ───────────────────────────────────────────────────────


class SearchWileyInput(BaseModel):
    query: str = Field(description="Search query for scientific articles")
    start_year: Optional[int] = Field(
        default=None, description="Start year filter (inclusive)"
    )
    end_year: Optional[int] = Field(
        default=None, description="End year filter (inclusive)"
    )
