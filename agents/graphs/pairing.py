"""Record-level checks for the groundedness verifier: is a value attached to the right thing?

``grounding.py`` checks that every value in a reply *exists* in the tool data. That cannot see
"S2C_43PGQ has 1.16% cloud" when 1.16% belongs to S2C_43PHQ, or "the clearest scene is X" when
another scene is clearer. This module adds two deterministic checks (no LLM) on top:

1. **Pairing.** The tool results are turned into *records*: a scene (STAC item, embedding) keyed by
   its id, a weather day keyed by its date. A line or sentence of the reply that names exactly one
   record may only carry decimal numbers (and, for scenes, dates) that belong to that record. A
   number that exists on *another* record but not on this one is flagged.
2. **Superlatives.** "clearest / lowest cloud / hottest / wettest / most similar ..." next to one
   record is checked against the min or max of that field over the same tool result.

Deliberately conservative, because a false alarm on a correct answer costs more than a miss:
- only decimal numbers are paired (small integers collide with counts and tile numbers);
- a segment naming 0 or 2+ records, or comparing ("than", "vs", "while", ...), is not paired
  strictly: with 2+ records a number may belong to any of them;
- numbers that match no record at all are left to the existence check in ``grounding.py``.

Limits: a swap *within* a segment that names several records, a record referred to only by
position ("the second one"), and claims with no value in them are not checked.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ID_KEYS = ("id", "item_id", "embedding_id")
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
MONTH_DAY_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})\b", re.I)
COMPARISON_RE = re.compile(r"\b(than|vs\.?|versus|compared|whereas|while|against|but)\b|[<>≤≥]|\d\s*[–—]\s*\d", re.I)
HEDGE_BEFORE_RE = re.compile(r"\b(among|amongst|one of|second|third|next|top)\W+(?:\w+\W+){0,2}$", re.I)
MONTH_DAY_RANGE_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})\s*[–—-]\s*(\d{1,2})\b", re.I)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z*(])")

# (pattern, fields to test (any one suffices), "min" | "max")
SUPERLATIVES: List[Tuple[re.Pattern, Tuple[str, ...], str]] = [
    (re.compile(r"\b(clearest|least cloud\w*|lowest cloud\w*|minimum cloud\w*)\b", re.I), ("cloud_cover",), "min"),
    (re.compile(r"\b(cloudiest|most cloud\w*|highest cloud\w*|maximum cloud\w*)\b", re.I), ("cloud_cover",), "max"),
    (re.compile(r"\b(hottest|warmest)\b", re.I), ("temperature_2m_max",), "max"),
    (re.compile(r"\b(coldest|coolest)\b", re.I), ("temperature_2m_max", "temperature_2m_min"), "min"),
    (re.compile(r"\b(wettest|rainiest|most (?:rain|precip\w*))\b", re.I), ("precipitation_sum",), "max"),
    (re.compile(r"\b(windiest|breeziest|strongest wind)\b", re.I), ("windspeed_10m_max",), "max"),
    (re.compile(r"\b(calmest|least wind\w*)\b", re.I), ("windspeed_10m_max",), "min"),
    (re.compile(r"\b(most similar|closest)\b", re.I), ("cosine_similarity",), "max"),
    (re.compile(r"\b(least similar|most different|furthest)\b", re.I), ("cosine_similarity",), "min"),
]


class Record:
    """One thing the tools described: a scene/embedding (by id) or a weather day (by date)."""

    def __init__(self, alias: str):
        self.aliases: Set[str] = {alias}
        self.nums: List[float] = []
        self.dates: Set[str] = set()
        self.fields: Dict[int, Dict[str, float]] = {}  # tool-output index -> {field: value}

    def add_nums(self, group: int, fields: Dict[str, float]) -> None:
        self.fields.setdefault(group, {}).update(fields)
        for v in fields.values():
            self._add(v)

    def _add(self, v: float) -> None:
        self.nums.append(v)
        if abs(v) <= 1:
            self.nums.append(v * 100)  # a ratio reported as a percentage
        if v < 0:
            self.nums.append(abs(v))  # "1.289°S" for latitude -1.289


def _scalar_numbers(obj: Any, prefix: str = "") -> Iterable[Tuple[str, float]]:
    """Scalar numeric leaves (not lists) with their field name, "eo:cloud_cover" -> "cloud_cover"."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict,)):
                yield from _scalar_numbers(v, str(k))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                yield str(k).split(":")[-1], float(v)
    return


def _dates_in(obj: Any) -> Set[str]:
    out: Set[str] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _dates_in(v)
    elif isinstance(obj, str):
        m = re.match(r"\d{4}-\d{2}-\d{2}", obj)
        if m and len(obj) >= 10:
            out.add(m.group(0))
    return out


def build_records(parsed_outputs: List[Any]) -> Dict[str, Record]:
    """alias -> Record, merged across tool outputs (the same scene seen by two tools is one record)."""
    by_alias: Dict[str, Record] = {}

    def register(aliases: List[str]) -> Record:
        existing = [by_alias[a] for a in aliases if a in by_alias]
        rec = existing[0] if existing else Record(aliases[0])
        for other in existing[1:]:  # merge records that turned out to be the same thing
            if other is rec:
                continue
            rec.aliases |= other.aliases
            rec.nums += other.nums
            rec.dates |= other.dates
            for g, f in other.fields.items():
                rec.fields.setdefault(g, {}).update(f)
        rec.aliases |= set(aliases)
        for a in rec.aliases:
            by_alias[a] = rec
        return rec

    def walk(obj: Any, group: int) -> None:
        if isinstance(obj, dict):
            daily = obj.get("daily")
            if isinstance(daily, dict) and isinstance(daily.get("time"), list):
                times = [str(t) for t in daily["time"]]
                for i, day in enumerate(times):
                    fields = {
                        k: float(v[i]) for k, v in daily.items()
                        if k != "time" and isinstance(v, list) and len(v) == len(times)
                        and isinstance(v[i], (int, float)) and not isinstance(v[i], bool)
                    }
                    if fields:
                        rec = register([day])
                        rec.dates.add(day)
                        rec.add_nums(group, fields)
            aliases = [obj[k] for k in ID_KEYS if isinstance(obj.get(k), str) and obj.get(k)]
            if aliases:
                fields = dict(_scalar_numbers(obj))
                if fields:
                    rec = register(aliases)
                    rec.dates |= _dates_in(obj)
                    rec.add_nums(group, fields)
            for v in obj.values():
                walk(v, group)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, group)

    for g, parsed in enumerate(parsed_outputs):
        walk(parsed, g)
    return by_alias


def _segments(text: str) -> List[str]:
    out: List[str] = []
    for line in text.replace("**", "").splitlines():
        line = line.strip()
        if not line or re.fullmatch(r"[|\-\s:]+", line):
            continue
        out += [s for s in SENTENCE_SPLIT_RE.split(line) if s.strip()]
    return out


def _entities(segment: str, by_alias: Dict[str, Record], day_dates: Set[str]) -> List[Record]:
    found: List[Record] = []
    for alias, rec in by_alias.items():
        if alias in segment and rec not in found:
            found.append(rec)
    pairs = [(_MONTHS[m.group(1)[:3].lower()], int(m.group(2))) for m in MONTH_DAY_RE.finditer(segment)]
    for m in MONTH_DAY_RANGE_RE.finditer(segment):  # "Jan 13-15" names every day in the range
        month = _MONTHS[m.group(1)[:3].lower()]
        pairs += [(month, d) for d in range(int(m.group(2)), int(m.group(3)) + 1)]
    for month, day in pairs:  # "Jan 22" -> the one weather day with that month/day
        hits = [d for d in day_dates if int(d[5:7]) == month and int(d[8:10]) == day]
        if len(hits) == 1 and by_alias[hits[0]] not in found:
            found.append(by_alias[hits[0]])
    return found


def _kind(rec: Record) -> str:
    """A weather day (keyed by date) and a scene (keyed by id) never explain each other's numbers."""
    return "day" if any(re.fullmatch(r"\d{4}-\d{2}-\d{2}", a) for a in rec.aliases) else "scene"


def _holds(token: str, vals: List[float]) -> bool:
    n = float(token)
    d = len(token.split(".")[1]) if "." in token else 0
    return any(round(v, d) == round(n, d) or abs(v - n) < 10 ** (-d) for v in vals)


def _label(rec: Record) -> str:
    return sorted(rec.aliases, key=lambda a: (a.startswith("emb_"), len(a), a))[0]


def check_pairing(
    text: str,
    scrub,
    tool_outputs_parsed: List[Any],
    user_numbers: Optional[List[float]] = None,
) -> List[str]:
    """Issues for values attached to the wrong record, and false superlatives. *scrub* removes
    dates, ids and product names from a segment so only numbers remain (from grounding.py)."""
    by_alias = build_records(tool_outputs_parsed)
    if not by_alias:
        return []
    records = list({id(r): r for r in by_alias.values()}.values())
    day_dates = {a for a, r in by_alias.items() if re.fullmatch(r"\d{4}-\d{2}-\d{2}", a)}
    user_numbers = user_numbers or []
    issues: List[str] = []

    for seg in _segments(text):
        ents = _entities(seg, by_alias, day_dates)
        if not ents:
            continue
        union = [v for r in ents for v in r.nums]
        decimals = re.findall(r"(?<![\w.])-?\d+\.\d+(?![\w])", scrub(seg))
        # "X (18.01% cloud) and 2.79% on 2024-01-10" names a second scene by date only: not one record
        joins_two = bool(re.search(r"\band\b", seg, re.I)) and len(set(decimals)) >= 2
        strict = len(ents) == 1 and not COMPARISON_RE.search(seg) and not joins_two

        for tok in dict.fromkeys(t for t in re.findall(r"(?<![\w.])-?\d+\.\d+(?![\w])", scrub(seg))):
            if float(tok) == 0 or _holds(tok, union) or _holds(tok, user_numbers):
                continue  # 0.0 is on nearly every record (precipitation, nodata), so it proves nothing
            owners = [r for r in records if r not in ents and _kind(r) == _kind(ents[0]) and _holds(tok, r.nums)]
            if owners and strict:
                issues.append(f"pairing {tok}: not a value of {_label(ents[0])}, it belongs to {_label(owners[0])}")

        is_scene = not any(re.fullmatch(r"\d{4}-\d{2}-\d{2}", a) for a in ents[0].aliases)
        if strict and is_scene and ents[0].dates:
            for d in dict.fromkeys(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", seg)):
                if d not in ents[0].dates and any(d in r.dates for r in records if r is not ents[0]):
                    issues.append(f"pairing {d}: not the date of {_label(ents[0])}")

        if strict:
            issues += _superlative_issues(seg, ents[0], records)
    return list(dict.fromkeys(issues))


def _superlative_issues(seg: str, ent: Record, records: List[Record]) -> List[str]:
    out: List[str] = []
    for pattern, fields, direction in SUPERLATIVES:
        word = pattern.search(seg)
        if not word or HEDGE_BEFORE_RE.search(seg[: word.start()]):  # "among the clearest", "second clearest"
            continue
        ok_any, best_desc, tested = False, None, False
        for fld in fields:
            for g, mine in ent.fields.items():
                if fld not in mine:
                    continue
                pool = [(r, r.fields[g][fld]) for r in records if g in r.fields and fld in r.fields[g]]
                if len(pool) < 2:
                    continue
                tested = True
                pick = min if direction == "min" else max
                target = pick(v for _, v in pool)
                if abs(mine[fld] - target) <= 1e-9 * max(1.0, abs(target)):
                    ok_any = True
                elif best_desc is None:
                    leader = next(r for r, v in pool if v == target)
                    best_desc = f"{_label(leader)} has the {'lowest' if direction == 'min' else 'highest'} {fld} ({target:g})"
        if tested and not ok_any:
            out.append(f"pairing '{word.group(0)}': {_label(ent)} is not it; {best_desc}")
    return out
