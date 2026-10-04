"""Deterministic groundedness check — no LLM.

Claim: every concrete value in the agent's reply (ISO dates, scene IDs,
numbers) must be traceable to a tool result in the session, or to the user's
own messages. Anything left over is an *unsupported claim*.

What counts as "supported" for numbers, since a reply legitimately rounds and
summarises:
- the number appears in a tool result, to the precision the reply used
  (17.3601 supports "17.36" and "17");
- a ratio in [-1, 1] stated as a percentage (0.73 -> 73%);
- it is a count, min, max, mean or sum of a numeric list in a tool result
  ("7 scenes", "max 31.2°C", "average 24.5");
- the user said it (echoed "10%", "2024").

Beyond existence, ``pairing.py`` checks that a value sits with the right record (a scene's cloud
cover, a weather day's temperature) and that "clearest / hottest / most similar" claims are true.

Known limits (documented, not hidden): a hallucinated value that happens to
coincide with some other number *on the same record* is not caught, a derived
value outside count/min/max/mean/sum (a difference, a ratio) is flagged even
when correct, and pairing only covers lines that name exactly one record (a swap
inside a sentence comparing two scenes, or a reference by position, is not
checked). It is a cheap tripwire, not a proof.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Iterable, List

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?Z?)?")
ISO_DAY_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
# S2B_31UDQ_20240105_0_L2A, LC09_L2SP_144048_20240112_...: >=2 underscore-joined tokens
SCENE_ID_RE = re.compile(r"\b[A-Z0-9]{2,}(?:_[A-Za-z0-9]+){2,}\b")
PRODUCT_RE = re.compile(r"\b[A-Za-z]{3,}-\d+[A-Za-z]?\b")
NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")


def _flatten(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flatten(v)
    else:
        yield obj


def _numeric_lists(obj: Any) -> Iterable[List[float]]:
    """Yield every list of plain numbers found anywhere in obj (e.g. a daily series)."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _numeric_lists(v)
    elif isinstance(obj, (list, tuple)):
        nums = [x for x in obj if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if nums and len(nums) == len(obj):
            yield [float(x) for x in nums]
        for v in obj:
            yield from _numeric_lists(v)


def _as_obj(text: str) -> Any:
    """Parse a tool result. The tools node stores ``str(result)``, a Python repr with single
    quotes, so JSON alone would miss most results and the series stats (sum/min/max/mean) with them."""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return text


def _date_parts(text: str) -> List[float]:
    """Year, month and day of every ISO date in text ("Jan 28" derives from 2024-01-28)."""
    return [float(int(p)) for m in ISO_DAY_RE.findall(text) for p in m]


def _supported_numbers(tool_outputs: List[str], user_texts: List[str]) -> List[float]:
    vals: List[float] = []
    for out in tool_outputs:
        obj = _as_obj(out)
        for leaf in _flatten(obj):
            if isinstance(leaf, bool):
                continue
            if isinstance(leaf, (int, float)):
                vals.append(float(leaf))
                if leaf < 0:
                    vals.append(abs(float(leaf)))  # "1.289°S" is how a reply writes latitude -1.289
                if isinstance(leaf, float) and abs(leaf) <= 1:
                    vals.append(leaf * 100)  # a ratio (similarity 0.73) is legitimately reported as 73%
        # numbers embedded in strings (display names, "count": "7", error text)
        vals += [float(n) for n in NUM_RE.findall(out)]
        vals += _date_parts(out)
        # counts of every list/dict level, and stats for numeric series
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, (list, dict)):
                vals.append(float(len(cur)))
                stack.extend(cur.values() if isinstance(cur, dict) else cur)
        for series in _numeric_lists(obj):
            vals += [min(series), max(series), sum(series), sum(series) / len(series)]
    for text in user_texts:
        vals += [float(n) for n in NUM_RE.findall(text)] + _date_parts(text)
    return vals


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def _number_supported(token: str, vals: List[float]) -> bool:
    n = float(token)
    d = _decimals(token)
    # equal after rounding to the reply's precision, or off by less than one unit in
    # the last place (covers truncation as well as rounding)
    return any(round(v, d) == round(n, d) or abs(v - n) < 10 ** (-d) for v in vals)


def _strip_list_markers(reply: str) -> str:
    """Drop numbered-list markers ("1. ") and markdown-table row-index cells ("| 5 |")."""
    text = re.sub(r"^\s*\d+[.)]\s+", "", reply, flags=re.M)
    return re.sub(r"^(\s*\|)\s*\d+\s*\|", r"\1", text, flags=re.M)


# Sentences that offer or suggest something ("try <=10% cloud", "e.g. 2024-02-29", "would you like me to ...")
# state values the agent is *proposing*, not results it is claiming, so those values need no source. Only
# dates and numbers are exempt, never scene ids, and a value that also appears in a plain claim is not.
ADVISORY_RE = re.compile(
    r"(?<!\w)e\.g\.|(?<!\w)i\.e\.|\b(?:for example|for instance|such as|try|trying|suggest\w*|consider|would you like|"
    r"do you want|want me to|if you(?:'d| would)? (?:like|want)|let me know|perhaps|maybe|relax\w*)\b", re.I)


def _advisory_exempt(text: str, scrub) -> set:
    """Dates and numbers that appear only in advisory sentences of *text*."""
    from .pairing import _segments

    advisory, plain = set(), set()
    for seg in _segments(text):
        tokens = {d[:10] for d in DATE_RE.findall(seg)} | set(NUM_RE.findall(scrub(seg)))
        (advisory if ADVISORY_RE.search(seg) else plain).update(tokens)
    return advisory - plain


def check_groundedness(
    reply: str,
    tool_outputs: List[str],
    user_texts: List[str],
    mode: str = "strict",
    allow: Iterable[str] = (),
    capability_texts: Iterable[str] = (),
) -> dict:
    """Return {"checked": n, "unsupported": [...], "grounded": bool}.

    mode "strict": dates, scene IDs and numbers must all be supported.
    mode "ids_dates": numbers are not checked. Use it where the reply
    legitimately offers example thresholds ("try <=10% cloud") that are advice,
    not claims about results; dates and scene IDs must still be sourced.
    ``allow``: exact values (e.g. a corrected date the reply suggests) to ignore.
    ``capability_texts``: the bound tools' descriptions and schemas. Whole numbers in them ("16 days ahead",
    "back to 1940") are facts about what the tools can do, which a reply may quote; decimals still need a
    tool result.

    With no tool outputs, any date/scene ID/number in the reply is unsupported
    unless the user supplied it.
    """
    allowed = {str(a) for a in allow}
    scrub_values = lambda t: PRODUCT_RE.sub(" ", SCENE_ID_RE.sub(" ", DATE_RE.sub(" ", t)))  # noqa: E731
    capability_texts = list(capability_texts)
    haystack = "\n".join(tool_outputs + user_texts)
    text = _strip_list_markers(reply.replace("\u2212", "-"))  # typographic minus -> "-"
    unsupported: List[str] = []
    checked = 0
    exempt = _advisory_exempt(text, scrub_values)

    # Timestamps are checked by their date part; the time-of-day is not verified.
    dates = [m[:10] for m in DATE_RE.findall(text)]
    for d in dict.fromkeys(dates):
        checked += 1
        if d not in haystack and d not in allowed and d not in exempt:
            unsupported.append(f"date {d}")

    ids = [i for i in SCENE_ID_RE.findall(text) if not DATE_RE.fullmatch(i)]
    for sid in dict.fromkeys(ids):
        checked += 1
        if sid not in haystack and sid not in allowed:
            unsupported.append(f"id {sid}")

    if mode == "strict":
        # product names ("Sentinel-2", "Landsat-8") are labels, not values
        scrubbed = PRODUCT_RE.sub(" ", SCENE_ID_RE.sub(" ", DATE_RE.sub(" ", text)))
        vals = _supported_numbers(tool_outputs, user_texts)
        vals += [float(n) for t in capability_texts for n in NUM_RE.findall(t) if "." not in n]  # whole numbers only
        for tok in dict.fromkeys(NUM_RE.findall(scrubbed)):
            checked += 1
            if tok not in allowed and tok not in exempt and not _number_supported(tok, vals):
                unsupported.append(f"number {tok}")

        # is each value attached to the right scene / day, and are "clearest", "hottest", ... true?
        from .pairing import check_pairing

        user_numbers = [float(n) for t in user_texts for n in NUM_RE.findall(t)]
        scrub = lambda s: PRODUCT_RE.sub(" ", SCENE_ID_RE.sub(" ", DATE_RE.sub(" ", s)))  # noqa: E731
        for issue in check_pairing(text, scrub, [_as_obj(o) for o in tool_outputs], user_numbers):
            checked += 1
            if issue not in allowed:
                unsupported.append(issue)

    return {"checked": checked, "unsupported": unsupported, "grounded": not unsupported}


def _cited_as_source(reply: str, name: str) -> bool:
    """True if *name* appears in parentheses attached to a value, e.g. "31.2 (get_weather)" or
    "(get_weather, 2024-01-01)". A tool merely named (in backticks, or "the tool (get_weather) reaches 16
    days") is a mention, not a claim that a result came from it."""
    n = re.escape(name)
    for m in re.finditer(rf"\(([^()\n]*\b{n}\b[^()\n]*)\)", reply):
        before = reply[max(0, m.start() - 60): m.start()].split("\n")[-1]
        if re.search(r"\d", m.group(1)) or re.search(r"\d|[A-Z0-9]{2,}_\w+", before):
            return True
    return False


def find_uncalled_tool_citations(reply: str, tool_names: Iterable[str], called: Iterable[str]) -> List[str]:
    """Tool names the reply cites as the source of a value that were never called in the session."""
    called_set = set(called)
    return [f"tool {n} cited but never called" for n in tool_names if n not in called_set and _cited_as_source(reply, n)]
