# MCP server: setup, tools and how the agent uses them

Detail for the *MCP server* section of the [README](../README.md). The TerraMind side is in
[terramind-details.md](terramind-details.md); how the API reaches both is in [API_MCP_A2A_interaction.md](API_MCP_A2A_interaction.md).

The MCP server (`mcp_server/server.py`, FastMCP, streamable-HTTP transport) is the one place the agent gets
geography, imagery and weather from. It exposes 5 tools. Each calls a free public API that needs no key. The agent
never imports these functions: it reaches them only through an MCP connection, which is what the assignment asks for.

## Run it

```bash
python -m mcp_server.server          # http://127.0.0.1:8765/mcp  (also started by scripts/start_all.sh)
```

| Variable | Default | Meaning |
|---|---|---|
| `MCP_SERVER_HOST`, `MCP_SERVER_PORT` | `127.0.0.1`, `8765` | where it listens |
| `MCP_SERVER_URL` | `http://127.0.0.1:8765/mcp` | where the API looks for it (API side, `service/mcp_client.py`) |

The API cannot start without it: the MCP connection is the one required tool source.

## The five tools

Every parameter has a description written for the model (FastMCP builds the JSON Schema from the function signature;
each parameter is `Annotated[T, Field(description=...)]`). A bare type hint would give the model only "array of
number"; the descriptions say, for example, "get this from `geocode_location` first if the user named a place".

| Tool | Required | Optional (default) | Calls | Notes |
|---|---|---|---|---|
| `geocode_location` | `query` | `limit` (1, max 5) | Nominatim (OpenStreetMap) | Returns `best_match` with `lat`, `lon`, `bbox` `[west, south, east, north]`. "No matches found" is an error result |
| `list_stac_collections` | none | none | Earth Search STAC | Collection id, title and description |
| `search_stac_items` | `bbox` | `start_date`, `end_date` (YYYY-MM-DD), `collections` (Sentinel-2 L2A), `max_cloud_cover` (0-100), `limit` (10, max 50) | Earth Search STAC | Returns `count` and items with id, datetime, cloud cover, bbox, assets, thumbnail. Cloud filter applies to optical collections only |
| `get_stac_item` | `collection`, `item_id` | none | Earth Search STAC | Full item: properties, geometry, asset links |
| `get_weather` | `lat`, `lon` | `start_date`, `end_date`, `daily_variables` (max/min temperature, precipitation, max wind) | Open-Meteo | No dates: today plus a 7-day forecast. Past dates go to the archive (back to 1940), recent or future ones to the forecast API (up to 16 days ahead) |

Endpoints: `nominatim.openstreetmap.org`, `earth-search.aws.element84.com/v1`, `archive-api.open-meteo.com` and
`api.open-meteo.com`. HTTP calls time out after 10 seconds. The tools are plain `_impl` functions in
`agents/tools/` (`geocoding.py`, `stac.py`, `weather.py`) that `server.py` wraps with `@mcp.tool()`. The same
functions also have LangChain `@tool` wrappers, used only by `scripts/test_tier1_agent.py` for a local smoke test and
never by the served agent.

The tool descriptions chain the tools together ("resolve place names to a bbox with `geocode_location` first"), which
is how the model learns the order geocode → search → get item, with no tool names in the system prompt.

## How the agent connects

At start-up, `service/mcp_client.py` opens a `MultiServerMCPClient` on the server's URL
(`transport: streamable_http`) and calls `get_tools()`. Each MCP tool becomes a LangChain tool, which
`wrap_tool_with_tracing` wraps so every call is timed and logged to `logs/traces.jsonl` with the session id. During a
turn the tools node calls them like any other tool; each call is an MCP `tools/call` over HTTP. `logs/mcp.log` shows
a short MCP session per call (open, request, close).

## How failures look

There are two kinds, and the agent handles both without crashing:

| Kind | Example | What comes back over MCP |
|---|---|---|
| **A tool returns an error value** | `get_stac_item` with an unknown id; `search_stac_items` with a 3-number bbox; Nominatim or Earth Search unreachable; a weather range outside both Open-Meteo windows | A normal result: `{"error": "Item 'NOPE' not found in collection 'sentinel-2-l2a'"}`. The model reads it like any result |
| **A tool raises** | `get_weather` with `start_date="2024-02-30"` (`date.fromisoformat` fails) | An MCP error result (`isError: true`): `Error executing tool get_weather: day is out of range for month` |

Both examples above were run against the server. In the second kind the API's tools node also catches any exception
and returns `Tool error: ...` to the model, so one bad call never ends the turn or the process. The first kind is
not logged as an error in the trace (the call succeeded); the second kind is, with the message in `error`. The
assignment's forced-error demo uses the second kind: turn 3 of [`demo/demo_transcript.md`](../demo/demo_transcript.md),
whose trace has `"error": "Error executing tool get_weather: day is out of range for month"`.

## Check it by hand

List the tools and call one with the MCP Python client (the same package the server uses):

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://127.0.0.1:8765/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print([t.name for t in (await s.list_tools()).tools])
            res = await s.call_tool("geocode_location", {"query": "Hyderabad"})
            print(res.content[0].text)

asyncio.run(main())
```

Expected tool list: `['geocode_location', 'list_stac_collections', 'search_stac_items', 'get_stac_item', 'get_weather']`.
Through the API, `GET /health` shows the same five names (plus the TerraMind skills when that agent is up) and
`GET /tools` shows each tool's description as the model sees it.

## Limits

- **No authentication.** The server listens on localhost and the upstream APIs need no keys. Production EO services
  (CDSE, SentinelHub, openEO) would need OAuth and key handling.
- **Geocoding returns one match by default**, so an ambiguous name ("Springfield") resolves silently to the first hit.
  `limit` up to 5 returns more candidates, but the agent does not yet ask the user to choose.
- **Public APIs are rate limited and can change.** Nominatim's usage policy applies (the server sends a
  `User-Agent`); tool-level tests against the three APIs are not mocked, and a hard-coded value in a live eval can go
  stale when an upstream revises its history (see `evals/SCENARIOS.md`).
- **No tool-result prompt-injection test.** Tool output is trusted as data; only injection in the user's message is
  tested (`prompt_injection_user`).
