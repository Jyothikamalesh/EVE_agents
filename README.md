# EVE Tier 1 EO Agent — LangGraph + MCP take-home

A single LangGraph ReAct agent for Earth Observation queries — geocode a
place, search satellite imagery (STAC), get weather — exposed over HTTP,
calling its tools through a real MCP server, with per-session conversation
state, bounded context, and a readable trace of every tool call.

## Install & run (clean checkout)

Needs Python 3.11–3.13 and a free Groq key (console.groq.com). One environment, no conda:

```bash
scripts/setup.sh                 # venv + dependencies + .env + TerraMind weights (~1 GB total)
#   scripts/setup.sh --skip-terramind   core agent only: no torch, much smaller
$EDITOR .env                     # GROQ_API_KEY=gsk_...
scripts/start_all.sh             # MCP server + TerraMind agent + API; Ctrl-C stops all
python scripts/run_demo.py       # the three required turns
```

Or start each process yourself (venv activated):

```bash
python -m mcp_server.server              # 1. MCP server: geocoding, STAC, weather   :8765
python -m terramind_agent.server         # 2. TerraMind A2A agent (optional)          :8767
uvicorn service.api:app --port 8000      # 3. API: the LangGraph agent                :8000
streamlit run scripts/ui_app.py          # 4. optional UI
```

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'content-type: application/json' \
  -d '{"session_id": "s1", "message": "Imagery of Hyderabad, January 2024, under 10% cloud?"}'
```

`requirements.lock` pins the exact versions this was tested with (`setup.sh` installs with it as a
constraint file; it was frozen on Python 3.13 / macOS arm64).

If the TerraMind agent isn't running the API logs a warning and starts without its
three tools. Why one environment works: the old setup used conda only for GDAL, but
`rasterio` 1.4.x wheels bundle GDAL, so `pyproject.toml` caps `rasterio<1.5` (1.5 ships
no wheels) and `pip install -e ".[dev,terramind]"` is enough. No OpenMP workaround
variables are needed on macOS arm64 / Python 3.13.

## Models

The agent only needs a LangChain chat model with tool binding; `service/llm.py` is the one place
that picks it, from environment variables (`.env.example` lists them):

| `EVE_LLM_PROVIDER` | serves | default endpoint | needs |
|---|---|---|---|
| `groq` (default) | hosted open-weight (default `qwen/qwen3.8-27b`) | Groq | `GROQ_API_KEY` |
| `ollama` | local open-weight (default model `qwen3:8b`) | `http://localhost:11434/v1` | nothing |
| `vllm` | self-hosted open-weight | `http://localhost:8000/v1` | `EVE_MODEL` |
| `openrouter` | many open-weight models | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY`, `EVE_MODEL` |
| `hf` | Hugging Face router / endpoints | `https://router.huggingface.co/v1` | `HF_TOKEN`, `EVE_MODEL` |
| `openai` | any other OpenAI-compatible server | set `EVE_LLM_BASE_URL` | key, `EVE_MODEL` |

Examples: `EVE_LLM_PROVIDER=ollama EVE_MODEL=qwen3:8b` or
`EVE_LLM_PROVIDER=vllm EVE_MODEL=Qwen/Qwen3-8B EVE_LLM_BASE_URL=http://gpu-box:8000/v1`.
`EVE_LLM_BASE_URL` / `EVE_LLM_API_KEY` override any preset. Everything except Groq goes through
the OpenAI-compatible chat API, which is what Ollama, vLLM, OpenRouter and HF all serve.
Models that don't emit native tool calls still work: the graph parses text-format calls
(`[TOOL_CALLS]…`, JSON arrays and the XML `<tool_use>` / `<function=…>` styles,
`agents/graphs/utils.py`). Tested: the Groq path end to end (evals) and the OpenAI-compatible path
(live, against Groq's compatible endpoint, plus offline config tests); I have **not** run it
against a real Ollama, vLLM, OpenRouter or HF server here. Tool-calling quality varies with the
model: that is the first thing to check when switching.

## Architecture

```
┌─────────────┐  HTTP (streamable-http, MCP protocol)   ┌──────────────────┐
│ mcp_server/  │◄─────────────────────────────────────── │ service/api.py   │
│  server.py   │   5 tools: geocode_location,             │  FastAPI, /chat  │
│  (FastMCP)   │   list/search/get_stac_item, get_weather │  POST            │
└──────┬───────┘                                          └────────┬─────────┘
       │ calls                                                     │ builds
       ▼                                                            ▼
agents/tools/*.py          service/eo_agent/graph.py (EOReactAgent) ──┐
 _impl functions:                subclasses agents/graphs/react      │
 geocoding, stac, weather         .ReactAgent — same tool-calling     │
 (httpx / pystac-client,          loop, scoped system prompt          │
  no API keys)                                                       │
                                                                       ▼
                                                        LangGraph StateGraph
                                                        (agent ⇄ tools loop)
```

**Why two layers of tool definitions.** `agents/tools/*.py` holds the actual
API-calling code as plain `_impl` functions (`geocode_location_impl`,
`search_stac_items_impl`, ...). Two different front doors call them:
LangChain `@tool` wrappers (used only by `scripts/test_tier1_agent.py` for quick
local smoke-testing, never by the graph that's actually served) and `mcp_server/server.py`'s `@mcp.tool()` wrappers (what
the served agent actually uses). This satisfies the assignment's "importing
the tool functions straight into the graph does not count" — `service/api.py`
gets its tools exclusively from `MultiServerMCPClient.get_tools()`
(`service/mcp_client.py`), an MCP client talking to the server process over
HTTP, not a Python import of `agents.tools`.

## The graph

`service/eo_agent/graph.py::EOReactAgent` is `agents/graphs/react/graph.py
::ReactAgent` with one override: its own `prompts.yaml` (scoped to the tool
families this service has, with explicit groundedness and citation rules).

```
START → context → agent ⇄ tools
                    └──(final answer)──→ verify → END
```

- **`context`** — rolling summary (see Context policy). Does nothing except
  every `EVE_SUMMARY_EVERY` turns.
- **`agent`** — calls the LLM (bound to the MCP tools) with the instruction, the
  summary, and the compacted/trimmed history. Tool calls (native, or the
  text-format `[TOOL_CALLS]` variants handled by
  `agents/graphs/utils.py::parse_text_tool_calls`) route to `tools`; a plain
  answer routes to `verify`.
- **`tools`** — runs each requested call (MCP-sourced, tracing-wrapped), turns
  any exception into a `ToolMessage` (`"Tool error: …"`) instead of raising,
  and loops back to `agent`.
- **`verify`** — checks the final answer against the tool results (see
  Hallucination control). Runs once per turn, outside the tool loop.
- **`agent_fallback`** — not wired here (no `fallback_llm`); the library
  supports it.

## Context policy

Three layers, each separate from storage:

1. **Stored per session:** the full LangGraph message list, keyed by
   `thread_id = session_id` in an `AsyncSqliteSaver` (`data/checkpoints.sqlite`).
   Nothing is dropped from storage; it survives an API restart (verified by
   killing `uvicorn` and asking a follow-up in a fresh process). Single-box
   durability; `AsyncPostgresSaver` is the same `thread_id` interface for
   multi-host.
2. **Tool-output compaction** (`agents/graphs/context.py`,
   `service/eo_agent/compaction.py`): when the prompt is built, tool results
   from *earlier* turns are replaced by a per-tool compact form — a STAC
   search keeps only `id / date / cloud` per scene, weather keeps min/max/mean
   with their dates, geocoding keeps name, lat/lon, bbox. Tool messages carry
   a `[compacted; call the tool again for full detail]` note. The current turn
   is never compacted and the stored history is untouched (the UI, evals and
   audits still see the raw payloads).
3. **Rolling summary:** after every `EVE_SUMMARY_EVERY` (default 3) turns that
   have aged out of the last `EVE_KEEP_RECENT_TURNS` (default 2) verbatim
   turns, one LLM call folds them into a running summary (facts only: places,
   bbox, dates, filters, scene IDs, errors) and they leave the prompt. With the
   defaults the summary fires at turns 6, 9, … A failed summary call keeps the
   old summary; the 96k-token `trim_messages` window is the final backstop.

**Demonstrated:** `demo/demo_transcript_context.md` / `demo/demo_trace_context.jsonl`
(`python scripts/run_demo_context.py`) — a 6-turn session where the summary
fires (`context_summary` trace step) and a later follow-up that only the summary
can answer still resolves correctly. On that session the prompt history went from
~9,100 to ~1,400 tokens (about 85% smaller). `evals/test_context_policy.py`
covers the mechanics offline.

## Hallucination control and evals

- **Prompt rules:** every factual claim must come from a tool result; tool
  errors are reported, not papered over; and each value is cited with its
  source tool, e.g. `cloud cover 3% (search_stac_items, S2C_43PGQ_…)`. Without
  document retrieval, that provenance is what "citation" means here.
- **`verify` node** (`agents/graphs/verify.py`, `agents/graphs/grounding.py`):
  a deterministic check (no LLM) that every date, scene ID and number in the
  final answer is traceable to a tool result, the rolling summary, or the
  user's own words (with rounding, counts and min/max/mean/sum allowed), and
  that every tool the answer cites was actually called. On failure the model
  gets one rewrite pass with this turn's tool results and the offending
  values; if values still don't trace, they are listed in a visible
  `⚠️ Could not verify…` caveat. Every turn adds a `verify` step to the trace.
  Known limit: a wrong value that coincides with another number in the
  payload isn't caught; it is a tripwire, not a proof.
- **Eval set** (`evals/`, 16 cases: tool use, multi-turn context, refusal,
  robustness, error path, hallucination): `python -m evals.run_evals`
  scores tool use, reply content and groundedness, and writes
  `evals/results/<stamp>.{json,md}`. Offline tests:
  `python -m evals.test_groundedness`, `test_verify`, `test_context_policy`,
  `test_terramind_skills`, `test_loop_guard`, `test_text_tool_calls`, `test_llm_factory`.
  See `evals/README.md`.

## Session memory

The agent's memory of a conversation is the checkpointed state plus the
rolling summary above. Cross-session semantic memory is deliberately not part
of the core design.

## Sessions, logs and traceability

Everything is keyed by the **session id** (the `session_id` you send to `/chat`):

- `GET /sessions/{id}` returns the stored conversation (tool calls and results included);
  404 if unknown. `GET /sessions/{id}/export` returns one JSON file with the messages, the API
  trace for that session, the TerraMind agent's log lines and the metadata of the embeddings it
  created. `python scripts/export_session.py <id>` writes it to a file.
- The UI loads a session by id (`Load session`) or replays an exported file read-only
  (`Open a session file`, no API history needed), so a session JSON can be sent to someone who
  can then read the whole test. `demo/session_terramind_example.json` is one such file.
- Logs: every API log line carries `[session_id]`, so `grep <id> logs/api.log`; per-tool-call trace
  lines are in `logs/traces.jsonl` (`jq 'select(.session_id=="<id>")'`); the TerraMind agent writes
  `logs/terramind.jsonl` (one line per skill call: session, skill, args, ok/error, duration,
  embedding id) and `[session_id]` on its console log. The session id travels to the TerraMind
  agent as the A2A message's `context_id` and metadata.
- TerraMind embeddings are persisted in `data/embeddings/<embedding_id>.npz` + `.json` (the JSON
  records the creating session), so they survive an agent restart (verified: restart, then
  `compare_embeddings` on earlier ids). Ids are scene-keyed, so any session can reuse them.

## MCP server (section 2.3)

`mcp_server/server.py` — `FastMCP`, `streamable-http` transport, 5 tools:
`geocode_location`, `list_stac_collections`, `search_stac_items`,
`get_stac_item`, `get_weather`. Each calls a free, no-auth-key public API
(Nominatim/OSM, Earth Search/Element84, Open-Meteo). `service/mcp_client.py`
connects via `langchain_mcp_adapters.client.MultiServerMCPClient` and that
connection — not a direct import — is the only way `service/api.py` obtains
tools for the graph.

## Errors, logging, trace (section 2.4)

- **Loop guard:** a model that keeps retrying a failing tool is stopped after `EVE_MAX_TOOL_CALLS`
  (default 10) calls in a turn: pending calls are answered as skipped and the turn ends with a plain
  message including the last real error, instead of running to the recursion limit.

- A failing tool **never crashes the process**: `agents/graphs/base.py::
  AgentGraph.make_tools_node` wraps every tool call in `try/except` and
  returns `ToolMessage(content=f"Tool error: {exc}")`, so the ReAct loop
  (and the FastAPI process) keeps running — the model sees the failure and
  explains it instead of the client getting a 500.
- Every tool call is additionally wrapped for tracing
  (`service/tracing.py::wrap_tool_with_tracing`), timed, and logged —
  success or failure — as one JSON line to `logs/traces.jsonl`:
  `{timestamp, session_id, step, tool_name, args, duration_ms, error,
  final_answer}`. `step="tool_call"` per tool invocation, `step="agent_turn"`
  once per `/chat` request (total latency + final reply). Every `/chat`
  response also echoes that request's trace steps inline (`trace` field) so
  a reply can be matched to its trace without grepping the log file.
- **Demo turn forcing a tool error:** `demo/demo_transcript.md` turn 3 —
  `get_weather` called with `start_date='2024-02-30'` (not a real calendar
  date). `date.fromisoformat` raises `ValueError: day is out of range for
  month` inside `agents/tools/weather.py`; it propagates through the MCP
  server, `make_tools_node` catches it, the model explains the failure in
  plain language, and `GET /health` immediately after confirms the server
  stayed up. Full trace: `demo/demo_trace.jsonl`.

## HTTP API (section 2.5)

FastAPI, `service/api.py`. `POST /chat {"session_id", "message"} ->
{"session_id", "reply", "trace"}`; `GET /health`. Run with `uvicorn
service.api:app --port 8000` (see above).

## TerraMind agent over A2A (bonus)

A second, separate **agent** (`terramind_agent/`, official `a2a-sdk` 1.x, A2A protocol 1.0, JSON-RPC)
wraps the **TerraMind-1.0-tiny** EO foundation model (IBM/ESA, Apache 2.0). It publishes an Agent
Card at `/.well-known/agent-card.json` with three skills, and the EO agent calls them as ordinary
tools (`service/a2a_client.py`; skills are discovered from the card):

| skill / tool | what it does |
|---|---|
| `embed_scene(collection, item_id, patch_size=224)` | fetches the 6 Sentinel-2 bands over one shared geographic window, runs TerraMind, keeps the full (196, 192) tensor **server-side** under an `embedding_id`; returns the id, scene date, cloud cover, tile id, crop bbox, shape and stats |
| `compare_embeddings(id_a, id_b)` | cosine similarity of the mean-pooled embeddings; when both scenes share a tile id also per-tile mean/min/max and the 3 least-similar grid cells (row, col) = *where* they differ; says `same_footprint: false` otherwise |
| `rank_similar(reference_id, candidate_ids)` | ranks embedded scenes by similarity to a reference |

The LLM only ever sees ids and summary numbers, never the tensor, and the verifier checks the
numbers it quotes like any other tool output. The skills are deterministic, so the remote agent is
thin; making its executor LLM-driven would not change the card or the wire format. Embeddings live
in an in-memory dict (process lifetime); a vector store is the next step.

Guards: a crop that is more than 20% no-data (a scene at the swath edge; I hit this live: two
different March scenes embedded to *identical* zero-input vectors with similarity 1.0) is rejected
with an explanation instead of embedded; the 20 m SWIR bands are read over the same ground window as
the 10 m bands (an earlier version read the same *pixel* window, which covers twice the ground at
20 m). Demo: `python scripts/run_demo_terramind.py` writes `demo/demo_transcript_terramind.md` and
the matching trace. This replaced an earlier TerraMind MCP server, so there is now a single MCP server.

## What I left out, and what I'd do next

Scoped to the required core (2.1–2.6) plus the TerraMind A2A bonus — "small and
working beats big and half-working." Specifically cut:

- **Agent Skills (bonus B).** Not attempted. A2A (bonus A) is done for the TerraMind agent only;
  the EO agent itself is not exposed over A2A, and there is no supervisor or multi-agent routing.
- **Persistent checkpointer — done, partially.** Swapped `InMemorySaver` for
  `AsyncSqliteSaver`; state now survives a restart (verified: killed the API
  process, restarted it, a follow-up on the same session recalled an earlier
  fact with no reminder). That's as far as SQLite should go, though — it's a
  single-box fix (WAL mode handles multiple workers on one machine, not
  multiple hosts). For an actual distributed deployment I'd still move to
  `AsyncPostgresSaver` (`langgraph-checkpoint-postgres`) — same `thread_id`-
  keyed interface, so it's a one-line swap in `service/api.py` whenever
  that's warranted, not a redesign — and the natural fit since EVE's backend
  is already Mongo/Postgres-adjacent infrastructure.
- **Auth.** Explicitly Tier 1 / no-auth per the original tool-selection brief
  this assignment grew out of — no API keys needed for Nominatim, Earth
  Search, or Open-Meteo. Production EO integrations (per the related JD:
  CDSE, SentinelHub, openEO) need OAuth/key handling this doesn't cover.
- **Broader EO stack (openEO, CDSE, SentinelHub, GEE).** Named in the
  longer-term JD, not in this assignment's required scope. Tier 1
  (geocoding + STAC + weather) plus the TerraMind bonus tool prove the
  pattern for both "data catalogue" and "foundation model" integrations;
  these are the natural next additions once auth is in place.
- **Image rendering in the API response itself.** `scripts/ui_app.py`
  (Streamlit, a thin client of the API) renders STAC thumbnails
  inline; the API returns thumbnail URLs as plain JSON
  fields rather than fetching/embedding images, which is the right call for
  an API response but means a raw `curl` won't show pictures.
- **Automated test suite.** Offline tests cover the groundedness checker, the
  verifier node and the context policy (`python -m evals.test_*`, no network);
  the eval set runs live against the API and model. There's no CI wiring, and
  tool-level tests (Nominatim / Earth Search / Open-Meteo) are not mocked.
- **Hard guarantees on hallucination.** The verifier is a deterministic
  tripwire plus one rewrite; an LLM judge or NLI check would catch more
  semantic errors at the cost of another model call per turn.

## Repo layout

```
agents/                  Portable LangGraph library (no backend imports)
  graphs/
    base.py              AgentGraph, make_tools_node (catches tool errors)
    context.py           turns, tool-output compaction, rolling summary
    verify.py            answer verifier node (grounding.py = the checker)
    utils.py             text-format tool-call parsing (Mistral + JSON-array)
    react/graph.py       ReactAgent — the tool-calling loop
  tools/                 _impl functions + LangChain @tool wrappers
    geocoding.py  stac.py  weather.py
mcp_server/
  server.py              MCP server (FastMCP, streamable-http) — section 2.3
terramind_agent/         bonus: TerraMind A2A agent (skills.py = logic, server.py = A2A)
service/                 The one app wiring library + MCP + A2A + HTTP together
  eo_agent/              EOReactAgent — ReactAgent + scoped system prompt
  mcp_client.py          MCP client wiring
  a2a_client.py          A2A client: TerraMind skills as tools
  memory.py              Qdrant memory experiment, NOT connected
  tracing.py             per-tool-call timing/logging -> logs/traces.jsonl
  eo_agent/compaction.py per-tool compaction rules
  api.py                 FastAPI app — section 2.5
scripts/
  setup.sh start_all.sh  one-command setup / start everything
  run_demo.py            reproduces the required 3-turn demo
  run_demo_context.py    context-policy demo (summary fires)
  run_demo_terramind.py  TerraMind A2A demo
  demo_context_policy.py reproduces the context-bounding demo
  test_tier1_agent.py    manual CLI smoke test (direct-import tools, not MCP)
  ui_app.py              Streamlit UI: thin client of the API (sessions, maps, export/replay)
  export_session.py      write one session as a JSON file
evals/                   eval set, groundedness checker re-export, offline tests
demo/
  demo_transcript.md             the 3 required turns, replies matched to trace
  demo_trace.jsonl               raw trace from that run
  demo_transcript_terramind.md   bonus: real TerraMind embedding demo
  demo_trace_terramind.jsonl     raw trace from that run
logs/
  traces.jsonl           live trace log (grows on every /chat request)
```

## Develop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Library internals (for reuse outside this assignment)

`agents` is also a standalone, pip-installable LangGraph library (no backend
imports) usable on its own — this is the shape it had before this service
was built on top of it, and the shape a consuming backend (e.g. EVE's main
backend) would still install:

```bash
pip install git+https://github.com/eve-esa/agents.git
```

Use via `AGENT_GRAPH_TYPE`:

- Short name `react` — ReAct loop with tools (default; what this service subclasses).
- Short name `simple` — single LLM node, **no tools** (smoke tests).
- Fully qualified: `agents.graphs.react.graph.ReactAgent`,
  `agents.graphs.simple.graph.SimpleChatAgent`.

Each graph ships `prompts.yaml` next to `graph.py`; `AgentGraph` fills
`prompts` from that file at init (`system` key = lead-in instruction).
`compile()` receives `history` + `summary` from the caller; the base
`format_history` serializes them to a text prefix ahead of `prompts['system']`.

Fault tolerance (`langgraph>=1.2`) built into `ReactAgent`:

- **LLM nodes** — `TimeoutPolicy`, an in-place `RetryPolicy` (`LLM_RETRY`)
  for transient failures, and an `error_handler` routing to a dedicated
  `agent_fallback` recovery node once retries are exhausted (registered only
  when the caller supplies `fallback_llm` to `compile()` — not used by this
  service).
- **Tool nodes** — failures become `ToolMessage` content for the ReAct loop
  to recover from (no node-level retry, which would re-invoke every tool
  call in the turn) — see "Errors, logging, trace" above.

`agents.tools.TIER1_TOOLS` is the direct-import tool list (geocoding, STAC,
weather) — useful for quick experiments via `scripts/test_tier1_agent.py`, but **not** what `service/api.py` uses (see
Architecture above for why).
