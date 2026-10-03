"""Weather and climate via Open-Meteo — free, no API key.

Open-Meteo splits "recent/future" data (``api.open-meteo.com/v1/forecast``,
which also serves up to 92 days of recent history via ``past_days``/
``start_date``) from deep historical reanalysis
(``archive-api.open-meteo.com/v1/archive``, back to 1940). ``get_weather``
picks the right backend automatically from the requested date range so the
agent only needs one tool.
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_TIMEOUT = 10.0

# Archive data lags a few days behind real time; forecast API covers the gap.
_ARCHIVE_LAG_DAYS = 5
_FORECAST_PAST_LIMIT_DAYS = 92
_FORECAST_FUTURE_LIMIT_DAYS = 16

_DEFAULT_DAILY = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
]


class WeatherInput(BaseModel):
    lat: float = Field(description="Latitude in decimal degrees (WGS84).")
    lon: float = Field(description="Longitude in decimal degrees (WGS84).")
    start_date: Optional[str] = Field(
        default=None,
        description="Start date, YYYY-MM-DD. Omit together with end_date for 'today + 7-day forecast'.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date, YYYY-MM-DD. Omit together with start_date for 'today + 7-day forecast'.",
    )
    daily_variables: Optional[List[str]] = Field(
        default=None,
        description=(
            "Open-Meteo daily variable names to fetch, e.g. "
            "['temperature_2m_max','precipitation_sum']. Defaults to max/min "
            "temperature, precipitation sum, and max wind speed."
        ),
    )


def _parse(d: Optional[str], fallback: date) -> date:
    if not d:
        return fallback
    return date.fromisoformat(d)


async def get_weather_impl(
    lat: float,
    lon: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    daily_variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Shared implementation — called by both the LangChain tool below and the MCP server."""
    today = date.today()
    start = _parse(start_date, today)
    end = _parse(end_date, today + timedelta(days=7))

    if start > end:
        return {"error": f"start_date {start} is after end_date {end}"}

    variables = daily_variables or _DEFAULT_DAILY
    archive_cutoff = today - timedelta(days=_ARCHIVE_LAG_DAYS)
    forecast_past_limit = today - timedelta(days=_FORECAST_PAST_LIMIT_DAYS)
    forecast_future_limit = today + timedelta(days=_FORECAST_FUTURE_LIMIT_DAYS)

    if end <= archive_cutoff:
        url, source = ARCHIVE_URL, "archive"
    elif start >= forecast_past_limit and end <= forecast_future_limit:
        url, source = FORECAST_URL, "forecast"
    else:
        return {
            "error": (
                f"Date range {start}..{end} spans more than the forecast API's "
                f"{_FORECAST_PAST_LIMIT_DAYS}-day lookback window and isn't fully "
                "in the archived past. Split into a historical call ending before "
                f"{archive_cutoff.isoformat()} and a separate recent/forecast call."
            )
        }

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(variables),
        "timezone": "auto",
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.error("Open-Meteo request failed (%s): %s", source, exc)
        return {"error": f"Weather request failed: {exc}", "source": source}

    return {
        "source": source,
        "lat": data.get("latitude", lat),
        "lon": data.get("longitude", lon),
        "timezone": data.get("timezone"),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": data.get("daily", {}),
        "daily_units": data.get("daily_units", {}),
    }


@tool("get_weather", args_schema=WeatherInput)
async def get_weather(
    lat: float,
    lon: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    daily_variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Get daily weather for a point: historical, recent, or forecast.

    Pass coordinates (use ``geocode_location`` first if the user named a
    place). With no dates, returns today plus a 7-day forecast. Historical
    requests route automatically to the long-range archive (back to 1940);
    recent/future requests route to the forecast API (up to 16 days ahead).
    """
    return await get_weather_impl(lat, lon, start_date, end_date, daily_variables)
