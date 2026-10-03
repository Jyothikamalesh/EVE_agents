"""Per-tool compaction for the EO tools (see ``agents/graphs/context.py``).

Each compactor turns a tool result into the facts a *later* turn could still need
("the second scene", "that city", "the same dates") and drops the bulk: asset
lists, geometry, thumbnail URLs, raw daily arrays. The model is told the result
was compacted and can call the tool again if it needs full detail.
"""

from typing import Any, Dict


def _rnd(v, n=2):
    return round(v, n) if isinstance(v, (int, float)) else v


def _geocode(r: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in r:
        return r

    def slim(x):
        return {k: x.get(k) for k in ("display_name", "lat", "lon", "bbox")}

    out: Dict[str, Any] = {"query": r.get("query"), "best_match": slim(r["best_match"])}
    others = (r.get("results") or [])[1:3]
    if others:
        out["other_matches"] = [slim(x) for x in others]
    return out


def _collections(r: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in r:
        return r
    return {"collections": [c.get("id") for c in r.get("collections", [])]}


def _search(r: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in r:
        return r
    return {
        "count": r.get("count"),
        "collections": r.get("collections"),
        "date_range": r.get("date_range"),
        "bbox": r.get("bbox"),
        "items": [
            {
                "id": it.get("id"),
                "date": (it.get("datetime") or "")[:10],
                "cloud": _rnd(it.get("cloud_cover")),
            }
            for it in r.get("items", [])
        ],
    }


def _item(r: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in r:
        return r
    props = r.get("properties") or {}
    return {
        "id": r.get("id"),
        "collection": r.get("collection"),
        "datetime": r.get("datetime"),
        "cloud_cover": _rnd(props.get("eo:cloud_cover")),
        "bbox": r.get("bbox"),
        "assets": sorted((r.get("assets") or {}).keys()),
    }


def _weather(r: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in r:
        return r
    daily = r.get("daily") or {}
    times = daily.get("time") or []
    stats: Dict[str, Any] = {}
    for var, vals in daily.items():
        if var == "time" or not isinstance(vals, list):
            continue
        pts = [(t, v) for t, v in zip(times, vals) if isinstance(v, (int, float))]
        if not pts:
            continue
        if len(pts) <= 7:  # short series: keep it whole
            stats[var] = dict(pts)
            continue
        lo, hi = min(pts, key=lambda p: p[1]), max(pts, key=lambda p: p[1])
        s = {"min": lo[1], "min_date": lo[0], "max": hi[1], "max_date": hi[0],
             "mean": round(sum(v for _, v in pts) / len(pts), 2)}
        if "sum" in var:
            s["total"] = round(sum(v for _, v in pts), 2)
        stats[var] = s
    return {
        "source": r.get("source"),
        "lat": r.get("lat"),
        "lon": r.get("lon"),
        "start_date": r.get("start_date"),
        "end_date": r.get("end_date"),
        "days": len(times),
        "daily": stats,
        "units": r.get("daily_units"),
    }


EO_COMPACTORS = {
    "geocode_location": _geocode,
    "list_stac_collections": _collections,
    "search_stac_items": _search,
    "get_stac_item": _item,
    "get_weather": _weather,
}
