# How the API, MCP and A2A talk to each other

Four processes, three protocols. This page follows one request through all of them. Related: the endpoint
reference is [api.md](api.md); the UI is [ui.md](ui.md).

## The processes

| Process | Port | Started by | Protocol it speaks | Code |
|---|---|---|---|---|
| UI (Streamlit) | 8501 | `scripts/start_all.sh` | HTTP, as a client of the API | `scripts/ui_app.py` |
| API + EO agent | 8000 | same | HTTP in; MCP and A2A out | `service/api.py` |
| MCP server | 8765 (`/mcp`) | same | MCP over streamable HTTP | `mcp_server/server.py` |
| TerraMind agent | 8767 | same | A2A 1.0, JSON-RPC | `terramind_agent/server.py` |

The LangGraph agent is **not** its own process. It lives inside the API process. The two outbound arrows are the
only way it touches the outside world:

```
 client (UI / curl / scripts)
        │  POST /chat {session_id, message}
        ▼
 ┌──────────────────────── API process (:8000) ─────────────────────────┐
 │  FastAPI  →  LangGraph:  context → agent ⇄ tools → verify            │
 │                                         │                            │
 │              tool list built at start-up from:                       │
 │                ① MultiServerMCPClient.get_tools()   (required)       │
 │                ② A2A Agent Card skills              (optional)       │
 └─────────────────────────┬───────────────────────┬────────────────────┘
                MCP (HTTP) │                       │ A2A (JSON-RPC)
                           ▼                       ▼
                 MCP server :8765          TerraMind agent :8767
                 5 tools                   3 skills
                    │                         │
        Nominatim · Earth Search · Open-Meteo │ Earth Search (Sentinel-2 bands)
                                              └→ TerraMind model, embeddings on disk
```

## Start-up order, and why it matters

`scripts/start_all.sh` starts them in this order:

1. **MCP server.** The API cannot start without it. The script only sleeps 2 s for it (no readiness check).
2. **TerraMind agent.** Started and waited for (`GET /.well-known/agent-card.json`) *before* the API, because the
   API reads the Agent Card once, at start-up.
3. **API.** On start-up, `lifespan` in `service/api.py`:
   1. fetches the Agent Card and builds one tool per skill (`service/a2a_client.py::load_a2a_tools`). If the agent is
      unreachable it logs a warning and carries on with no TerraMind tools;
   2. connects to the MCP server and loads its tools (`service/mcp_client.py::load_traced_tools`). If this fails the
      API does not start;
   3. wraps every tool, MCP and A2A alike, for tracing, compiles the graph with the SQLite checkpointer, and records
      the tool names (shown by `GET /health`).
4. **UI.**

Because the tool list is fixed at start-up, starting the TerraMind agent *after* the API does not add its tools.
Restart the API. (Production would re-discover tools on a timer or on a failed call; see the README's "left out" list.)

## One request, end to end

User message: *"Embed the two clearest January scenes over Hyderabad and compare them."*

1. **Client to API.** `POST /chat {"session_id": "s1", "message": ...}`. The API stores `s1` in a context variable,
   so every log line and trace entry for this request carries it, then calls the graph with `thread_id = "s1"`.
2. **`context` node.** Loads `s1`'s stored messages from `data/checkpoints.sqlite`, compacts old tool results, and
   folds older turns into the summary if the prompt would pass the budget.
3. **`agent` node.** One LLM call. The model sees the tool list (names, descriptions, argument schemas) and answers
   with tool calls.
4. **`tools` node, MCP tools.** `geocode_location` then `search_stac_items`. Each call goes to the MCP server over
   HTTP, which calls the public API (Nominatim, Earth Search) and returns JSON. The agent never imports these
   functions; they are only reachable through the MCP connection.
5. **`tools` node, A2A tools.** `embed_scene` is, to the model, an ordinary tool. Inside, `call_skill` sends one A2A
   message to the TerraMind agent:
   ```json
   message.parts = [{"data": {"skill": "embed_scene", "args": {"collection": "sentinel-2-l2a", "item_id": "S2B_43QHV_20240113_0_L2A"}}}]
   message.context_id = "s1"                 # the chat session id
   message.metadata   = {"session_id": "s1"}
   ```
   The TerraMind agent replies with a **task**: it moves `SUBMITTED → WORKING → COMPLETED` and attaches an artifact
   whose data part is the skill's JSON result (`embedding_id`, scene date, cloud cover, tile id, `crop_bbox`, stats).
   The 196×192 tensor stays on the TerraMind side, saved in `data/embeddings/`; only the id travels back. A failed
   skill ends the task as `FAILED` with an error message, which the client raises as `TerraMind agent: ...`.
6. **Back to `agent`.** The tool results are appended and the model decides again: more tools (`compare_embeddings`
   takes the two ids), or a final answer. This repeats until the model answers without a tool call or the loop guard
   (`EVE_MAX_TOOL_CALLS`) stops it.
7. **`verify` node.** Checks every date, scene id and number in the final answer against *all* tool results of the
   turn, MCP and A2A alike. One rewrite, then a caveat if it still fails.
8. **API to client.** `{session_id, reply, trace, turn}`. The checkpointer has already saved the new messages.

## How each link works

### API to MCP server

- **Discovery.** `MultiServerMCPClient({"eo_tools": {"url": MCP_SERVER_URL, "transport": "streamable_http"}})`, then
  `get_tools()`. The MCP server describes each tool (name, description, JSON Schema built from the function
  signature, with per-field descriptions). Those descriptions are what the model sees: the prompt itself names no
  tools.
- **Calls.** The model's tool call becomes an MCP `tools/call`. Exceptions inside a tool travel back as an MCP error
  and are turned into a `Tool error: ...` message by the tools node, so the loop continues.
- **Why MCP and not an import.** It is the assignment's rule, and it also keeps the tools in a separate process that
  can be restarted or replaced without touching the agent. `agents/tools/*.py` holds the real code as `_impl`
  functions; `mcp_server/server.py` wraps them with `@mcp.tool()`.

### API to TerraMind agent

- **Discovery.** `A2ACardResolver` fetches `/.well-known/agent-card.json`. The card lists three skills with
  descriptions that say when to use each tool and how to read its numbers. `tools_from_card` makes one tool per
  skill, using the card's description as the tool description.
- **Argument schemas live in the client** (`_ARG_SCHEMAS`), because an Agent Card describes skills but cannot declare
  typed parameters. A skill on the card with no schema in the client is skipped with a warning.
- **Transport.** JSON-RPC on `/`, non-streaming. One skill call is one message and one task.
- **Session link.** The chat session id travels as the message's `context_id` and `metadata.session_id`. That is how
  `logs/terramind.jsonl` and `data/embeddings/<id>.json` can be traced back to a chat session, and how
  `GET /sessions/{id}/export` collects them.

### Where the two meet: the same loop, the same checks

MCP tools and A2A skills are both just entries in the tool list. The agent loop, the tracing wrapper, the loop
guard, the error handling and the verifier treat them identically. The only difference the model sees is the source
of the description: the MCP server's schema, or the Agent Card.

## What is shared and what is not

| | Where it lives |
|---|---|
| Conversation state | API process, SQLite checkpointer, keyed by session id |
| Traces (`traces.jsonl`, `node_runs/`) | API process |
| Embeddings and the TerraMind log | TerraMind agent (`data/embeddings/`, `logs/terramind.jsonl`) |
| Tool definitions | MCP server (EO tools) and the Agent Card (TerraMind) |
| Model API key | API process only; the MCP and TerraMind processes need none |

## When one link is down

| Down | At start-up | During a run |
|---|---|---|
| MCP server | API refuses to start | the tool call fails, the model reports it, the API stays up |
| TerraMind agent | API starts without the three tools | calls fail as `TerraMind agent: ...` and are reported |
| Model provider | API starts (only the key is checked) | `POST /chat` returns 500 |
| An upstream API (Nominatim, Earth Search, Open-Meteo) | no effect | the MCP tool errors and the model reports it |
