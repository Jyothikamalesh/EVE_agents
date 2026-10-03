"""Standalone, key-free Tier 1 tools (no backend imports).

Each submodule exposes LangChain ``@tool``-decorated async callables backed
by free, no-auth public APIs:

- ``geocoding`` — Nominatim (OpenStreetMap) place name -> lat/lon/bbox.
- ``stac`` — STAC catalog search (Earth Search / Element84) for EO imagery.
- ``weather`` — Open-Meteo historical and forecast weather/climate data.

``TIER1_TOOLS`` collects all of them for convenient ``tools=`` wiring into a
graph's ``compile(...)`` call.
"""

from .geocoding import geocode_location
from .stac import get_stac_item, list_stac_collections, search_stac_items
from .weather import get_weather

TIER1_TOOLS = [
    geocode_location,
    list_stac_collections,
    search_stac_items,
    get_stac_item,
    get_weather,
]

__all__ = [
    "geocode_location",
    "list_stac_collections",
    "search_stac_items",
    "get_stac_item",
    "get_weather",
    "TIER1_TOOLS",
]
