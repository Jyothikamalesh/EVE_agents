"""Geocoding via Nominatim (OpenStreetMap) — free, no API key.

Usually the first tool call in an EO agent turn: users say "Hyderabad" or
"the Amazon basin" and downstream tools (STAC search, weather) need a
lat/lon point and/or a bounding box.

Nominatim's usage policy (https://operations.osmfoundation.org/policies/nominatim/)
requires a descriptive ``User-Agent`` and caps at ~1 request/second — fine for
single-turn agent lookups, not for bulk geocoding.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "eve-esa-agents/0.5 (https://github.com/eve-esa/agents)"
_TIMEOUT = 10.0


class GeocodeInput(BaseModel):
    query: str = Field(
        description=(
            "Place name to geocode, e.g. 'Hyderabad', 'Mount Kilimanjaro', "
            "'Lake Victoria', 'Paris, France'. Free text, as specific as the "
            "user gave it."
        )
    )
    limit: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Max number of candidate matches to return (ambiguous names return more).",
    )


def _format_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    # Nominatim bbox is [south, north, west, east] as strings.
    bbox_raw = raw.get("boundingbox")
    bbox = None
    if bbox_raw and len(bbox_raw) == 4:
        south, north, west, east = (float(v) for v in bbox_raw)
        # Reorder to [west, south, east, north] — the standard order STAC/geo
        # APIs (incl. the stac tools in this package) expect.
        bbox = [west, south, east, north]

    return {
        "display_name": raw.get("display_name"),
        "lat": float(raw["lat"]),
        "lon": float(raw["lon"]),
        "bbox": bbox,
        "type": raw.get("type"),
        "osm_class": raw.get("class"),
        "importance": raw.get("importance"),
    }


async def geocode_location_impl(query: str, limit: int = 1) -> Dict[str, Any]:
    """Shared implementation — called by both the LangChain tool below and the MCP server."""
    params = {"q": query, "format": "jsonv2", "limit": limit, "addressdetails": 0}
    headers = {"User-Agent": USER_AGENT}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(NOMINATIM_URL, params=params, headers=headers)
            resp.raise_for_status()
            results: List[Dict[str, Any]] = resp.json()
    except httpx.HTTPError as exc:
        logger.error("Nominatim geocoding failed for %r: %s", query, exc)
        return {"error": f"Geocoding request failed: {exc}", "query": query}

    if not results:
        return {"error": "No matches found", "query": query}

    formatted = [_format_result(r) for r in results]
    return {"query": query, "results": formatted, "best_match": formatted[0]}


@tool("geocode_location", args_schema=GeocodeInput)
async def geocode_location(query: str, limit: int = 1) -> Dict[str, Any]:
    """Resolve a place name to coordinates and a bounding box.

    Use this before any spatial tool (STAC imagery search, weather lookup)
    whenever the user refers to a place by name rather than coordinates.
    Returns a lat/lon point plus a ``bbox`` in ``[west, south, east, north]``
    order (degrees, WGS84) suitable for passing straight into
    ``search_stac_items`` or ``get_weather``.
    """
    return await geocode_location_impl(query, limit)
