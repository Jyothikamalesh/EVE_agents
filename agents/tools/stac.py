"""STAC catalog search via Earth Search (Element84) — free, no API key.

Earth Search (https://earth-search.aws.element84.com/v1) indexes
Sentinel-2, Sentinel-1, Landsat and other public EO collections on AWS and
needs no credentials to query or read the resulting COG assets.

These tools are sync (``pystac-client`` wraps blocking HTTP calls); LangChain
runs sync ``@tool`` functions in a thread when invoked via ``ainvoke``, so
they compose fine in the async ReAct loop without blocking the event loop.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pystac_client import Client
from pystac_client.exceptions import APIError

logger = logging.getLogger(__name__)

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
_DEFAULT_LIMIT = 10
_MAX_LIMIT = 50


def _open_client() -> Client:
    return Client.open(EARTH_SEARCH_URL)


def list_stac_collections_impl() -> Dict[str, Any]:
    """Shared implementation — called by both the LangChain tool below and the MCP server."""
    try:
        client = _open_client()
        collections = [
            {"id": c.id, "title": c.title, "description": c.description}
            for c in client.get_collections()
        ]
    except APIError as exc:
        logger.error("STAC collection listing failed: %s", exc)
        return {"error": f"Collection listing failed: {exc}"}

    return {"catalog": EARTH_SEARCH_URL, "collections": collections}


@tool("list_stac_collections")
def list_stac_collections() -> Dict[str, Any]:
    """List the EO imagery collections available to search (e.g. Sentinel-2, Landsat).

    Call this when unsure which ``collections`` value to pass to
    ``search_stac_items``, or when the user asks what imagery is available.
    """
    return list_stac_collections_impl()


class StacSearchInput(BaseModel):
    bbox: List[float] = Field(
        description=(
            "Bounding box [west, south, east, north] in decimal degrees "
            "(WGS84). Get this from geocode_location if the user named a place."
        )
    )
    start_date: Optional[str] = Field(
        default=None, description="Start of date range, YYYY-MM-DD."
    )
    end_date: Optional[str] = Field(
        default=None, description="End of date range, YYYY-MM-DD."
    )
    collections: Optional[List[str]] = Field(
        default=None,
        description=(
            "STAC collection IDs to search, e.g. ['sentinel-2-l2a']. Defaults "
            "to Sentinel-2 L2A (optical, atmospherically corrected) if omitted. "
            "Call list_stac_collections to see all options."
        ),
    )
    max_cloud_cover: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Maximum acceptable cloud cover percentage (0-100). Omit for no filter.",
    )
    limit: int = Field(
        default=_DEFAULT_LIMIT,
        ge=1,
        le=_MAX_LIMIT,
        description="Max number of items to return.",
    )


def _item_summary(item) -> Dict[str, Any]:
    props = item.properties
    return {
        "id": item.id,
        "collection": item.collection_id,
        "datetime": props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "bbox": item.bbox,
        "assets": sorted(item.assets.keys()),
        "thumbnail": _thumbnail_href(item),
    }


def _thumbnail_href(item) -> Optional[str]:
    asset = item.assets.get("thumbnail") or item.assets.get("visual")
    return asset.href if asset is not None else None


def search_stac_items_impl(
    bbox: List[float],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    collections: Optional[List[str]] = None,
    max_cloud_cover: Optional[float] = None,
    limit: int = _DEFAULT_LIMIT,
) -> Dict[str, Any]:
    """Shared implementation — called by both the LangChain tool below and the MCP server."""
    if len(bbox) != 4:
        return {"error": "bbox must be [west, south, east, north]"}

    collections = collections or ["sentinel-2-l2a"]
    datetime_range = None
    if start_date or end_date:
        datetime_range = f"{start_date or '..'}/{end_date or '..'}"

    query = None
    if max_cloud_cover is not None:
        query = {"eo:cloud_cover": {"lt": max_cloud_cover}}

    try:
        client = _open_client()
        search = client.search(
            collections=collections,
            bbox=bbox,
            datetime=datetime_range,
            query=query,
            max_items=limit,
            limit=limit,
        )
        items = list(search.items())
    except APIError as exc:
        logger.error("STAC search failed: %s", exc)
        return {"error": f"STAC search failed: {exc}"}

    return {
        "catalog": EARTH_SEARCH_URL,
        "collections": collections,
        "bbox": bbox,
        "date_range": datetime_range,
        "count": len(items),
        "items": [_item_summary(i) for i in items],
    }


@tool("search_stac_items", args_schema=StacSearchInput)
def search_stac_items(
    bbox: List[float],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    collections: Optional[List[str]] = None,
    max_cloud_cover: Optional[float] = None,
    limit: int = _DEFAULT_LIMIT,
) -> Dict[str, Any]:
    """Search for EO imagery scenes covering a bounding box and date range.

    Resolve place names to a ``bbox`` with ``geocode_location`` first. Use
    ``max_cloud_cover`` to filter cloudy optical scenes out (Sentinel-2 /
    Landsat only — ignored for radar collections like Sentinel-1).
    """
    return search_stac_items_impl(
        bbox, start_date, end_date, collections, max_cloud_cover, limit
    )


class StacItemInput(BaseModel):
    collection: str = Field(description="STAC collection ID the item belongs to, e.g. 'sentinel-2-l2a'.")
    item_id: str = Field(description="STAC item ID, as returned by search_stac_items.")


def get_stac_item_impl(collection: str, item_id: str) -> Dict[str, Any]:
    """Shared implementation — called by both the LangChain tool below and the MCP server."""
    try:
        client = _open_client()
        item = client.get_collection(collection).get_item(item_id)
    except APIError as exc:
        logger.error("STAC item lookup failed for %s/%s: %s", collection, item_id, exc)
        return {"error": f"Item lookup failed: {exc}"}

    if item is None:
        return {"error": f"Item {item_id!r} not found in collection {collection!r}"}

    return {
        "id": item.id,
        "collection": item.collection_id,
        "datetime": item.properties.get("datetime"),
        "properties": item.properties,
        "bbox": item.bbox,
        "geometry": item.geometry,
        "assets": {k: a.href for k, a in item.assets.items()},
    }


@tool("get_stac_item", args_schema=StacItemInput)
def get_stac_item(collection: str, item_id: str) -> Dict[str, Any]:
    """Get full metadata and asset links (including direct COG URLs) for one STAC item.

    Use after ``search_stac_items`` to inspect a specific scene — its assets
    carry the actual browsable/downloadable imagery URLs (e.g. ``visual``,
    ``thumbnail``, per-band COGs).
    """
    return get_stac_item_impl(collection, item_id)
