"""MCP server for Tier 1 EO tools — geocoding, STAC imagery search, weather.

This is the server referenced by assignment section 2.3: it exposes tools
over the Model Context Protocol (streamable-http transport) so the agent
calls them *through MCP*, not via direct Python import. The actual API
calls (Nominatim, Earth Search, Open-Meteo) live in ``agents.tools.*`` as
plain ``_impl`` functions shared with the LangChain-wrapped versions used
for local smoke testing (``scripts/test_tier1_agent.py``, ``ui_app.py``).

Every parameter is ``Annotated[T, Field(description=...)]``, not a bare type
hint — FastMCP builds each tool's JSON Schema straight from the function
signature, and a bare hint produces a schema with no per-field description
(just a title), unlike the original LangChain ``args_schema`` Pydantic
models in ``agents/tools/*.py`` which do carry per-field descriptions. That
gap was real and measured: a model bound to the bare-hint version sees
"bbox: array of number" instead of "bbox: get this from geocode_location
first if the user named a place" — materially less guidance per call,
which showed up as fewer/less-confident tool calls in testing. Field
descriptions here are copied from the matching ``*Input`` class in
``agents/tools/*.py`` so both front doors document tools identically.

Run:
    source .venv/bin/activate
    python -m mcp_server.server
    # listens on http://127.0.0.1:8765/mcp by default; override with
    # MCP_SERVER_HOST / MCP_SERVER_PORT.
"""

import os
from typing import Annotated, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from agents.tools.geocoding import geocode_location_impl
from agents.tools.stac import (
    get_stac_item_impl,
    list_stac_collections_impl,
    search_stac_items_impl,
)
from agents.tools.weather import get_weather_impl

HOST = os.environ.get("MCP_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("MCP_SERVER_PORT", "8765"))

mcp = FastMCP("eve-tier1-eo-tools", host=HOST, port=PORT)


@mcp.tool()
async def geocode_location(
    query: Annotated[
        str,
        Field(
            description=(
                "Place name to geocode, e.g. 'Hyderabad', 'Mount Kilimanjaro', "
                "'Lake Victoria', 'Paris, France'. Free text, as specific as the "
                "user gave it."
            )
        ),
    ],
    limit: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            le=5,
            description="Max number of candidate matches to return (ambiguous names return more).",
        ),
    ] = 1,
) -> dict:
    """Resolve a place name to lat/lon and a [west,south,east,north] bbox.

    Call this before any spatial tool whenever the user names a place
    instead of giving coordinates.
    """
    return await geocode_location_impl(query, limit)


@mcp.tool()
def list_stac_collections() -> dict:
    """List EO imagery collections available to search (e.g. Sentinel-2, Landsat)."""
    return list_stac_collections_impl()


@mcp.tool()
def search_stac_items(
    bbox: Annotated[
        List[float],
        Field(
            description=(
                "Bounding box [west, south, east, north] in decimal degrees "
                "(WGS84). Get this from geocode_location if the user named a place."
            )
        ),
    ],
    start_date: Annotated[
        Optional[str], Field(default=None, description="Start of date range, YYYY-MM-DD.")
    ] = None,
    end_date: Annotated[
        Optional[str], Field(default=None, description="End of date range, YYYY-MM-DD.")
    ] = None,
    collections: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description=(
                "STAC collection IDs to search, e.g. ['sentinel-2-l2a']. Defaults "
                "to Sentinel-2 L2A (optical, atmospherically corrected) if omitted. "
                "Call list_stac_collections to see all options."
            ),
        ),
    ] = None,
    max_cloud_cover: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=0,
            le=100,
            description="Maximum acceptable cloud cover percentage (0-100). Omit for no filter.",
        ),
    ] = None,
    limit: Annotated[
        int, Field(default=10, ge=1, le=50, description="Max number of items to return.")
    ] = 10,
) -> dict:
    """Search for EO imagery scenes covering a bounding box and date range.

    Resolve place names to a bbox with geocode_location first. Use
    max_cloud_cover to filter cloudy optical scenes out (Sentinel-2 /
    Landsat only — ignored for radar collections like Sentinel-1).
    """
    return search_stac_items_impl(
        bbox, start_date, end_date, collections, max_cloud_cover, limit
    )


@mcp.tool()
def get_stac_item(
    collection: Annotated[
        str, Field(description="STAC collection ID the item belongs to, e.g. 'sentinel-2-l2a'.")
    ],
    item_id: Annotated[
        str, Field(description="STAC item ID, as returned by search_stac_items.")
    ],
) -> dict:
    """Get full metadata and asset links (including direct COG URLs) for one STAC item.

    Use after search_stac_items to inspect a specific scene — its assets
    carry the actual browsable/downloadable imagery URLs (e.g. visual,
    thumbnail, per-band COGs).
    """
    return get_stac_item_impl(collection, item_id)


@mcp.tool()
async def get_weather(
    lat: Annotated[float, Field(description="Latitude in decimal degrees (WGS84).")],
    lon: Annotated[float, Field(description="Longitude in decimal degrees (WGS84).")],
    start_date: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Start date, YYYY-MM-DD. Omit together with end_date for "
                "'today + 7-day forecast'."
            ),
        ),
    ] = None,
    end_date: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "End date, YYYY-MM-DD. Omit together with start_date for "
                "'today + 7-day forecast'."
            ),
        ),
    ] = None,
    daily_variables: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description=(
                "Open-Meteo daily variable names to fetch, e.g. "
                "['temperature_2m_max','precipitation_sum']. Defaults to max/min "
                "temperature, precipitation sum, and max wind speed."
            ),
        ),
    ] = None,
) -> dict:
    """Get daily weather for a point: historical, recent, or forecast.

    Pass coordinates (use geocode_location first if the user named a
    place). With no dates, returns today plus a 7-day forecast. Historical
    requests route automatically to the long-range archive (back to 1940);
    recent/future requests route to the forecast API (up to 16 days ahead).
    """
    return await get_weather_impl(lat, lon, start_date, end_date, daily_variables)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
