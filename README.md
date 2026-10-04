# EVE Earth-Observation Agent — LangGraph + MCP + A2A take-home

A single LangGraph ReAct agent for Earth Observation queries — geocode a
place, search satellite imagery (STAC), get weather, and (bonus) embed and compare scenes with
the TerraMind foundation model through a separate A2A agent. It is exposed over HTTP and calls
its tools through a real MCP server, with per-session conversation state, bounded context, a
groundedness check on every answer, and a readable trace of every tool call.

## Install & run (clean checkout)

Needs Python 3.11–3.13 and a free Groq key (console.groq.com). One environment, no conda:

```bash
scripts/setup.sh                 # venv + dependencies + .env + TerraMind weights (~1 GB total)
#   scripts/setup.sh --skip-terramind   core agent only: no torch, much smaller
$EDITOR .env                     # GROQ_API_KEY=gsk_...
scripts/start_all.sh             # MCP server + TerraMind agent + API (EO agent) + UI; Ctrl-C stops all
#   scripts/start_all.sh --skip-terramind   without the TerraMind agent
#   scripts/start_all.sh --no-ui            without the Streamlit UI (UI: http://127.0.0.1:8501)
python scripts/run_demo.py       # the three required turns
```

Or start each process yourself (venv activated):

```bash
python -m mcp_server.server              # 1. MCP server: geocoding, STAC, weather   :8765
python -m terramind_agent.server         # 2. TerraMind A2A agent (optional)          :8767
uvicorn service.api:app --port 8000      # 3. API: the LangGraph agent                :8000
streamlit run scripts/ui_app.py          # 4. UI                                      :8501
```

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'content-type: application/json' \
  -d '{"session_id": "s1", "message": "Imagery of Hyderabad, January 2024, under 10% cloud?"}'
```

`requirements.lock` pins the exact versions this was tested with (`setup.sh` installs with it as a
constraint file; it was frozen on Python 3.13 / macOS arm64).

If the TerraMind agent isn't running the API logs a warning and starts without its
three tools; the prompt never mentions tools, so nothing refers to the missing ones, and the two
`a2a` evals skip themselves (`GET /health` lists the bound tools). Why one environment works: the old setup used conda only for GDAL, but
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
::ReactAgent` with one override: its own `prompts.yaml`, which holds only the cross-tool policy
(never invent, report failures, cite the source tool, stay in scope). **It names no tools.** Each
tool describes itself: MCP tools through their schemas and descriptions, A2A skills through their
Agent Card, so what the agent is told always matches what is actually bound.

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

Four layers, each separate from storage:

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
3. **Rolling summary, on demand:** when the prompt for the new turn would exceed
   `EVE_SUMMARY_TOKEN_BUDGET` (default 3,000 tokens, calibrated against the current prompt: re-check it if the prompt changes), every turn older than the last
   `EVE_KEEP_RECENT_TURNS` (default 2) verbatim turns is folded into a running summary (facts only:
   places, bbox, dates, filters, scene IDs, errors) by one LLM call, and those turns leave the prompt.
   Short conversations never pay for a summary call, and at least 2 turns are folded per call so a
   tight budget cannot cost one call per turn (unless the prompt is 1.5x over budget). Set
   `EVE_SUMMARY_TOKEN_BUDGET=0` for a fixed schedule instead (`EVE_SUMMARY_EVERY`, default 3). A failed
   summary call keeps the old summary; the 96k-token `trim_messages` window is the final backstop.
4. **Request ledger:** a summary keeps facts but loses order, so "the very first thing
   I asked" went wrong (the model answered with the latest request). For every turn folded into
   the summary, the prompt also carries the user's own words, numbered and in order
   (`request_ledger` in `agents/graphs/context.py`). It is built in code from the stored
   messages, with no LLM, so it cannot drift or invent, and costs about 30 tokens per turn.

**Demonstrated:** `demo/demo_transcript_context.md` / `demo/demo_trace_context.jsonl`
(`python scripts/run_demo_context.py`) — a 7-turn session. The prompt grows from about 450 to 3,100
tokens over turns 1-5; at the start of turn 6 it would pass the 3,000-token budget, so turns 1-3 are
folded (`context_summary` trace step) and the prompt drops to about 2,500 tokens. Turn 7 then asks
about turn 1 ("which city, what cloud limit, what dates?"), which has left the prompt, and is answered
correctly. For scale, the raw stored history of that session is about 9,700 tokens (tiktoken count of
the stored messages; the prompt figures are the node trace's approximate count).

**Measured:** the same 7-turn conversation run 5 times answered turn 7 correctly **5 of 5** with the
ledger (and again 5 of 5 after switching to the on-demand trigger). Before the ledger it was **1 of 3**: the summary held the Hyderabad request, but the model
reported the most recent one (Nairobi, 30%) as the first. The verifier cannot catch that failure,
since every value in the wrong answer exists in the tool data. `evals/test_context_policy.py`
covers the mechanics offline.

## Hallucination control and evals

- **Prompt rules:** every factual claim must come from a tool result; tool
  errors are reported, not papered over; and each value is cited with its
  source tool, e.g. `cloud cover 3% (search_stac_items, S2C_43PGQ_…)`. Without
  document retrieval, that provenance is what "citation" means here.
- **`verify` node** (`agents/graphs/verify.py`, `agents/graphs/grounding.py`):
  a deterministic check (no LLM) that every date, scene ID and number in the
  final answer is traceable to a tool result, the rolling summary, or the
  user's own words (with rounding, counts and min/max/mean/sum allowed; whole
  numbers quoted from a tool's own description, like "16 days ahead", count as
  supported; values in advisory sentences such as "e.g." or "would you like" are
  exempt, scene ids never), and that every tool the answer cites as the source of
  a value (`31.2 (get_weather)`) was actually called. On failure the model
  gets one rewrite pass with this turn's tool results and the offending
  values; if values still don't trace, they are listed in a visible
  `⚠️ Could not verify…` caveat. Every turn adds a `verify` step to the trace.
  Beyond existence, `agents/graphs/pairing.py` checks that a value sits with the
  *right record* (a scene's cloud cover, a weather day's temperature) and that
  "clearest / hottest / wettest / most similar" claims are true, by recomputing the
  min or max from the tool data. It only judges lines that name exactly one record and
  skips comparisons and ranges, so a correct answer is not flagged for how it is worded.
  **Measured** on all 137 final replies stored from this project's runs: 0 false alarms
  (the 2 replies it flagged were real errors, e.g. the hottest day given as 2026-03-26 when
  the data puts 34.8 °C on 2026-03-27, and a 58.1% scene called "the lowest cloud cover" next to
  an 18.0% scene), and it caught 1,266 of 1,271 (99.6%) deliberately swapped values.
  A separate **detection benchmark** (`python -m evals.eval_detection`, 284 labelled replies on fresh
  synthetic data, predictions written before the first run) reports precision, recall and F1 per scenario:
  100% / 99% / 100% on attribution, ids, dates, superlatives and citations, but weak on plain numbers: a
  random wrong 1-decimal number passes 33% of the time against a dense weather payload (1% for 2 decimals
  in a sparse one), and small integers, embedding ids, echoed user claims and wrong values hidden in
  advice text are missed. See `evals/SCENARIOS.md`.
  Known limits: a swap inside a sentence that compares several records, a record referred to
  only by position ("the second one"), and claims with no value in them are not checked; it
  is a tripwire, not a proof.
- **Eval set** (`evals/`, 25 cases: tool use, multi-turn context (incl. a 7-turn session), refusal, prompt injection,
  robustness, error path, hallucination): `python -m evals.run_evals`
  scores tool use, reply content and groundedness (`--repeat N` runs each case N times and reports passes per case, flaky cases and a pass rate with a 95% interval), and writes
  `evals/results/<stamp>.{json,md}`. Offline tests:
  `python -m evals.test_groundedness`, `test_pairing`, `test_verify`, `test_context_policy`,
  `test_terramind_skills`, `test_loop_guard`, `test_text_tool_calls`, `test_llm_factory`,
  `test_node_trace`, `test_a2a_client`, `test_run_evals`, `test_detection`.
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
tools (`service/a2a_client.py`; skills **and their descriptions** are read from the card, so the remote
agent owns its own usage instructions; only the argument schemas stay in the client, because an Agent
Card cannot declare typed parameters):

| skill / tool | what it does |
|---|---|
| `embed_scene(collection, item_id, patch_size=224)` | fetches the 6 Sentinel-2 bands over one shared geographic window, runs TerraMind, keeps the full (196, 192) tensor **server-side** under an `embedding_id`; returns the id, scene date, cloud cover, tile id, crop bbox, shape and stats |
| `compare_embeddings(id_a, id_b)` | cosine similarity of the mean-pooled embeddings; when both scenes share a tile id also per-tile mean/min/max and the 3 least-similar grid cells (row, col) = *where* they differ; says `same_footprint: false` otherwise |
| `rank_similar(reference_id, candidate_ids)` | ranks embedded scenes by similarity to a reference |

**Explaining the result.** The embeddings are appearance similarity with no land-cover labels, and the
scores are compressed near 1.0, so every `compare_embeddings` / `rank_similar` result ships its own
`reading_guide` (never a percentage of "sameness"; the third decimal is noise; a low tile says
*where* appearance changed, not *what* changed), `reference_points` (measured: the same tile on
different dates 0.998 to 0.999, two different tiles in southern India about 0.986, southern India against
the Sahara 0.94 to 0.95; three scenes, indicative only), and per-comparison `caveats` and `days_apart` (a cloudy scene or a long
gap can lower similarity). The EO prompt only says to follow such guidance, so the wording
comes from the agent that knows what its numbers mean.

The LLM only ever sees ids and summary numbers, never the tensor, and the verifier checks the
numbers it quotes like any other tool output. The skills are deterministic, so the remote agent is
thin; making its executor LLM-driven would not change the card or the wire format. Embeddings are
held in memory and also written to `data/embeddings/` (see "Sessions, logs and traceability"), so
they survive an agent restart; a real vector store is the next step.

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
  server.py              MCP server (FastMCP, streamable-http) — section 2.3
terramind_agent/         bonus: TerraMind A2A agent (skills.py = logic, server.py = A2A)
service/                 The one app wiring library + MCP + A2A + HTTP together
  eo_agent/              EOReactAgent — ReactAgent + scoped system prompt
  mcp_client.py          MCP client wiring
  a2a_client.py          A2A client: TerraMind skills as tools
  node_trace.py          node-level trace recorder (callback handler) behind the UI graph
  sessions.py            session view/export
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
  demo_transcript_context.md     context policy: the rolling summary fires, a later follow-up still resolves
  demo_trace_context.jsonl       raw trace from that run
  demo_transcript_terramind.md   bonus: real TerraMind embedding demo
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
timeouts, retries and an optional fallback model) plus the pieces added for this assignment
(`context.py`, `verify.py`, `grounding.py`, the loop guard). `service/eo_agent/` subclasses it
with an EO system prompt. The `simple` graph (single LLM node, no tools) is kept from upstream
for smoke tests.
