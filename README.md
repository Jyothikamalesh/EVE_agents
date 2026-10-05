# EVE Earth-Observation Agent — LangGraph + MCP + A2A

A single LangGraph ReAct agent for Earth Observation queries — geocode a
place, search satellite imagery (STAC), get weather, and (optional) embed and compare scenes with
the TerraMind foundation model through a separate A2A agent. It is exposed over HTTP and calls
its tools through a real MCP server, with per-session conversation state, bounded context, a
groundedness check on every answer, and a readable trace of every tool call.

## Install & run (clean checkout)

Needs Python 3.11–3.13 and a free Groq key (console.groq.com). One environment, no conda:

```bash
scripts/setup.sh                 # venv + dependencies + .env + TerraMind weights (~1 GB total)
$EDITOR .env                     # GROQ_API_KEY=gsk_...
scripts/start_all.sh             # MCP server + TerraMind agent + API (EO agent) + UI; 
python scripts/run_demo.py       # the three required turns
```


`requirements.lock` pins the exact versions this was tested with (`setup.sh` installs with it as a
constraint file; it was frozen on Python 3.13 / macOS arm64).

If the TerraMind agent isn't running the API logs a warning and starts without its
three tools; the prompt never mentions tools, so nothing refers to the missing ones, and the two
`a2a` evals skip themselves (`GET /health` lists the bound tools). Why one environment works: the old setup used conda only for GDAL, but
`rasterio` 1.4.x wheels bundle GDAL, so `pyproject.toml` caps `rasterio<1.5` (1.5 ships
no wheels) and `pip install -e ".[dev,terramind]"` is enough. No OpenMP workaround
variables are needed on macOS arm64 / Python 3.13.


## Architecture

```
 Streamlit UI :8501 / curl / scripts
              │  POST /chat  (session_id)
              ▼
 ┌──────────────────────────────────────────────┐   data/checkpoints.sqlite
 │ service/api.py  (FastAPI :8000)              │◄─ full history per session id
 │  LangGraph:  context → agent ⇄ tools → verify│   (AsyncSqliteSaver, survives restart)
 └───────┬──────────────────────┬───────────────┘
         │ MCP (streamable-http)│ A2A (JSON-RPC, Agent Card discovered at start-up)
         ▼                      ▼
 ┌────────────────────┐   ┌────────────────────────────────────────┐
 │ mcp_server :8765   │   │ terramind_agent :8767  (separate agent)│
 │ 5 tools:           │   │ /.well-known/agent-card.json           │
 │  geocode_location  │   │ 3 skills, shown to the model as tools: │
 │  list_stac_collect.│   │  embed_scene, compare_embeddings,      │
 │  search_stac_items │   │  rank_similar   (TerraMind-1.0-tiny)   │
 │  get_stac_item     │   └──────────────────┬─────────────────────┘
 │  get_weather       │                      │ Sentinel-2 bands
 └─────────┬──────────┘                      ▼
           │ httpx / pystac-client     Earth Search STAC
           ▼
  Nominatim · Earth Search STAC · Open-Meteo   (no API keys)

 context node : tool-output compaction, token-bound rolling summary, code-built request ledger
 verify node  : checks the reply against tool results (MCP and A2A alike), one rewrite, then a caveat
```


**Why two layers of tool definitions.** `agents/tools/*.py` holds the actual
API-calling code as plain `_impl` functions (`geocode_location_impl`,
`search_stac_items_impl`, ...). Two different front doors call them:
LangChain `@tool` wrappers (used only by `scripts/test_tier1_agent.py` for quick
local smoke-testing, never by the graph that's actually served) and `mcp_server/server.py`'s `@mcp.tool()` wrappers (what
the served agent actually uses). The served agent never imports these functions: `service/api.py`
gets its tools exclusively from `MultiServerMCPClient.get_tools()`
(`service/mcp_client.py`), an MCP client talking to the server process over
HTTP, not a Python import of `agents.tools`.

## The graph

`service/eo_agent/graph.py::EOReactAgent` is `agents/graphs/react/graph.py
::ReactAgent` with one override: its own `prompts.yaml`, which holds only the cross-tool policy
(never invent, report failures, cite the source tool, stay in scope). **It names no tools.** Each
tool describes itself: MCP tools through their schemas and descriptions, A2A skills through their
Agent Card, so what the agent is told always matches what is actually bound.

```
START → context → agent ⇄ tools
                    └──(final answer)──→ verify → END
```

- **`context`** — builds the prompt view each turn: compacts old tool output, and folds older turns
  into a rolling summary only when the prompt would exceed the token budget (see Context policy).
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

**Where the loop stops:** when the agent answers without requesting a tool (the answer then goes to `verify`), or after
`EVE_MAX_TOOL_CALLS` tool calls (default 10), when the `limit` node skips the pending calls and ends the turn with a plain
message that says it stopped and quotes the last error.
- **`agent_fallback`** — not wired here (no `fallback_llm`); the library
  supports it.

## Context policy

The full conversation is always **stored**. Each turn, the model is sent a **smaller view** of it:

1. **Stored per session:** every message, keyed by session id in a SQLite checkpointer (`data/checkpoints.sqlite`). Nothing is deleted, and it survives an API restart. (For several hosts: `AsyncPostgresSaver`, same interface.)
2. **Old tool output is shrunk:** results from earlier turns are replaced by a short form (a scene search keeps only id / date / cloud). The current turn is never shrunk.
3. **Long chats get a summary:** if the prompt would pass 3,000 tokens, older turns are folded into a short summary by one LLM call. Short chats never pay for it.
4. **A numbered list of the user's own requests** is added for the folded turns. It is built in code, with no LLM, so "what did I ask first?" is answered from the user's real words and not from the summary.

A 96k-token trim sits behind all of this as a last resort. Settings: `EVE_SUMMARY_TOKEN_BUDGET`, `EVE_KEEP_RECENT_TURNS`. More: [docs/context-details.md](docs/context-details.md).

**Shown working:** a 7-turn session ([transcript](demo/demo_transcript_context.md); run it with `python scripts/run_demo_context.py`).

- By turn 6 the prompt would pass the budget, so turns 1-3 are folded into a summary and the prompt shrinks.
- Turn 7 asks about turn 1, which has left the prompt, and is answered correctly (5 of 5 runs).
- Without the numbered request list this failed (1 of 3): the model gave the latest request as the first.

## Hallucination control and evals

- **Prompt rules:** every fact must come from a tool result, errors are reported rather than hidden, and each value names its source tool, e.g. `cloud cover 3% (search_stac_items, S2C_43PGQ_…)`.
- **`verify` node** (`agents/graphs/verify.py`): a rule-based check, no LLM. Every date, scene id and number in the final answer must trace to a tool result, the summary or the user's own words, and a value must sit on the right record (right scene, right day). If not, the model gets one rewrite, then a visible `⚠️ Could not verify…` caveat. It is a tripwire, not a proof: claims with no value in them ("mostly clear") are not checked.
- **Measured:** a benchmark of 284 labelled replies (`python -m evals.eval_detection`) gives precision 100%, recall 89%, F1 94%. It is strong on attribution, ids, dates, superlatives and citations, and weak on plain numbers in dense data, echoed user claims and derived values.
- **Evals:** 25 live cases (`python -m evals.run_evals --repeat 5`) plus offline tests (`python -m evals.test_*`). `run_evals` also prints the benchmark's precision / recall / F1.

More: [docs/hallucination-details.md](docs/hallucination-details.md), [evals/README.md](evals/README.md), [evals/SCENARIOS.md](evals/SCENARIOS.md).



## TerraMind agent over A2A

A second, separate **agent** (`terramind_agent/`, official `a2a-sdk` 1.x, A2A protocol 1.0, JSON-RPC)
wraps the **TerraMind-1.0-tiny** EO foundation model (IBM/ESA, Apache 2.0). It publishes an Agent
Card at `/.well-known/agent-card.json` with three skills, and the EO agent calls them as ordinary
tools (`service/a2a_client.py`; skills **and their descriptions** are read from the card, so the remote
agent owns its own usage instructions; only the argument schemas stay in the client, because an Agent
Card cannot declare typed parameters):

| skill / tool | what it does |
|---|---|
| `embed_scene(collection, item_id, patch_size=224)` | fetches the 6 Sentinel-2 bands over one shared geographic window, runs TerraMind, keeps the full (196, 192) tensor **server-side** under an `embedding_id`; returns the id, scene date, cloud cover, tile id, crop bbox, shape and stats |
| `compare_embeddings(id_a, id_b)` | cosine similarity of the mean-pooled embeddings; when both scenes share a tile id also per-tile mean/min/max and the 3 least-similar grid cells (row, col) = *where* they differ; says `same_footprint: false` otherwise |
| `rank_similar(reference_id, candidate_ids)` | ranks embedded scenes by similarity to a reference |

**How to read the numbers.** The embeddings measure how similar two scenes *look*. They carry no land-cover labels, and scores sit very close to 1.0. So every `compare_embeddings` / `rank_similar` result comes with its own reading guide, reference scores and caveats (a cloudy scene or a long gap can lower similarity). The EO prompt only says to follow that guidance; the wording comes from the agent that knows what its numbers mean.

- The LLM sees only ids and summary numbers, never the embedding tensor. The verifier checks the numbers it quotes like any other tool output.
- Embeddings are kept in memory and also saved in `data/embeddings/`, so they survive a restart.
- A crop that is more than 20% no-data (a scene at the edge of the satellite swath) is rejected with an explanation instead of being embedded.

Demo: `python scripts/run_demo_terramind.py` ([transcript](demo/demo_transcript_terramind.md)). More: [docs/terramind-details.md](docs/terramind-details.md).

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

## Graph trace in the UI

After every turn the UI draws the graph for that turn (`🧭 Graph trace`): the fixed topology
(`context → agent ⇄ tools → verify`, plus the TerraMind A2A agent when it was called), with the path
that actually ran in blue and numbered in execution order, each node's run count and total time,
and a hover tooltip per node (tokens in, turns in the prompt, whether a summary block was included,
how many earlier tool results were compacted, tools requested, verifier verdict). Below it, "Inspect a
step" shows one node run: what it was given, each model call (latency, tokens, response, requested
tools), each tool call (args, result rendered like in the chat, remote agent and duration), what it
produced, and a **Full prompt** popover with exactly what the model was sent.

How it is recorded: `service/node_trace.py` is a LangChain callback handler passed in the run config
(the graph code is untouched). It records every node run in order with its model and tool calls and
is saved per turn in `logs/node_runs/<session_id>.jsonl` (prompts capped at 100 KB per call). The API
serves it at `GET /sessions/{id}/node_runs`, the session export includes it, so reloading a session by
id or replaying an exported file shows the same graphs. `python -m evals.test_node_trace` checks the
recorded sequence, model calls, tool calls and errors offline with a fake model.

## MCP server

`mcp_server/server.py` — `FastMCP`, `streamable-http` transport, 5 tools:
`geocode_location`, `list_stac_collections`, `search_stac_items`,
`get_stac_item`, `get_weather`. Each calls a free, no-auth-key public API
(Nominatim/OSM, Earth Search/Element84, Open-Meteo). `service/mcp_client.py`
connects via `langchain_mcp_adapters.client.MultiServerMCPClient` and that
connection — not a direct import — is the only way `service/api.py` obtains
tools for the graph.

## Errors, logging, trace

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

## HTTP API

FastAPI, `service/api.py`. `POST /chat {"session_id", "message"} ->
{"session_id", "reply", "trace"}`; `GET /health`. Run with `uvicorn
service.api:app --port 8000` (see above).


## What I left out, and what I'd do next

Scope is the agent, MCP tools, context policy, error handling and API, plus the TerraMind A2A agent. Deliberately left out:

- **Agent Skills (`SKILL.md`).** Not attempted. A2A is done for the TerraMind agent only;
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
- **Auth.** None: Nominatim, Earth Search and Open-Meteo need no API keys. Production EO
  services (CDSE, SentinelHub, openEO) need OAuth/key handling this doesn't cover.
- **Broader EO stack (openEO, CDSE, SentinelHub, GEE).** Not in
  scope here. Geocoding, STAC and weather plus the TerraMind agent prove the
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
- **TerraMind crops are not centred on the place.** `embed_scene` embeds a 2.2 km square at the centre of
  the Sentinel-2 tile (about 110 km across), not at the geocoded place; in my checks the crop was 25-28 km
  from Hyderabad and Bengaluru. The result's `note` and `crop_bbox` say so and the agent relays it, but the
  proper fix is an optional point or bbox argument that centres the crop on the place (and a footprint test
  by crop overlap instead of tile id), so "similar to Hyderabad" would really mean Hyderabad.
- **Verifier strictness on advisory text: found by `--repeat`, fixed.** The first 90-run measurement showed the
  verifier rewriting 20% of turns that were actually good (capability statements quoting a tool's own limits,
  suggested thresholds and dates, tool names merely mentioned), sometimes into a worse answer. Whole numbers from
  tool descriptions now count as supported, values in advisory sentences are exempt (never scene ids), and a tool
  name is a citation only when attached to a value; the rewrite prompt keeps explanations and suggestions. Now:
  75/75 runs over 25 cases, 4 of 102 turns flagged, 3 of those a genuine repeatable table-transcription error
  (see `evals/README.md`). Remaining limit: a derived value (a difference of two scores) is still flagged.
- **Hard guarantees on hallucination.** The verifier is a deterministic
  tripwire plus one rewrite: every value must exist in the tool data and sit
  with the right record, and superlatives are recomputed. It does not check
  claims with no value in them ("mostly clear skies"), swaps inside a sentence
  that compares records, or references by position. Next steps: a larger,
  independently written test set for the catch and false-alarm rates (the 137
  replies above are this project's own runs). An LLM judge or NLI check would
  catch more semantic errors at the cost of another model call per turn.

## Repo layout

```
agents/                  Portable LangGraph library (no backend imports)
  graphs/
    base.py              AgentGraph, make_tools_node (catches tool errors)
    context.py           turns, tool-output compaction, rolling summary
    verify.py            answer verifier node (grounding.py + pairing.py = the checker)
    utils.py             text-format tool-call parsing (Mistral + JSON-array)
    react/graph.py       ReactAgent — the tool-calling loop
  tools/                 _impl functions + LangChain @tool wrappers
    geocoding.py  stac.py  weather.py
mcp_server/
  server.py              MCP server (FastMCP, streamable-http)
terramind_agent/         TerraMind A2A agent (skills.py = logic, server.py = A2A)
service/                 The one app wiring library + MCP + A2A + HTTP together
  eo_agent/              EOReactAgent — ReactAgent + scoped system prompt
  mcp_client.py          MCP client wiring
  a2a_client.py          A2A client: TerraMind skills as tools
  node_trace.py          node-level trace recorder (callback handler) behind the UI graph
  sessions.py            session view/export
  tracing.py             per-tool-call timing/logging -> logs/traces.jsonl
  eo_agent/compaction.py per-tool compaction rules
  api.py                 FastAPI app
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
  demo_transcript_context.md     context policy: the rolling summary fires, a later follow-up still resolves
  demo_trace_context.jsonl       raw trace from that run
  demo_transcript_terramind.md   real TerraMind embedding demo
  demo_trace_terramind.jsonl     raw trace from that run
logs/
  traces.jsonl           live trace log (grows on every /chat request)
```

## Develop

```bash
scripts/setup.sh                    # or: python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,terramind]"
python -m evals.test_groundedness   # offline tests, no network or model (see evals/README.md)
python -m evals.run_evals           # live evals against the running API
```

## Reusing the agent library

`agents/` is the portable LangGraph library this service is built on (from
[eve-esa/agents](https://github.com/eve-esa/agents)): `ReactAgent` (the tool loop, with LLM
timeouts, retries and an optional fallback model) plus the pieces added for this project
(`context.py`, `verify.py`, `grounding.py`, the loop guard). `service/eo_agent/` subclasses it
with an EO system prompt. The `simple` graph (single LLM node, no tools) is kept from upstream
for smoke tests.
