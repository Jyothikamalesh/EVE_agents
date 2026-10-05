# Using the UI

The UI is `scripts/ui_app.py`, a Streamlit app on port 8501. It is a thin client of the HTTP API ([api.md](api.md)):
it holds no agent of its own, so what you see is exactly what the API does, with the same MCP tools, TerraMind
skills, verifier and stored sessions.

## Start it

```bash
scripts/setup.sh              # once: venv, dependencies, .env, TerraMind weights
$EDITOR .env                  # GROQ_API_KEY=...
scripts/start_all.sh          # MCP + TerraMind + API + UI, then open http://127.0.0.1:8501
```

Variants: `scripts/start_all.sh --skip-terramind` (core agent only, no embedding tools) and `--no-ui` (API only).
To run the UI alone against an API that is already running: `streamlit run scripts/ui_app.py`. It reads the API
address from `EVE_API_BASE` (default `http://127.0.0.1:8000`).

If the page loads but every message fails with "Request failed", the API is not up: check `logs/api.log`.

## Chatting

Type in the box at the bottom of a tab. Each tab is one **session**, and its id is also the `session_id` sent to
`POST /chat`. Follow-ups work as in the API: "what was the weather like there during the same period?" resolves from
the stored session.

Try these (they are also listed in the sidebar under **Try**):

1. *Show me cloud-free Sentinel-2 imagery of Hyderabad from January 2024.*
2. *What's the historical weather in Paris for the first week of July 2023?*
3. *Find two Sentinel-2 scenes over Hyderabad in January 2024, embed both with TerraMind and compare them.*
   (Takes about a minute: each embedding takes about 30 seconds.)

To see the error path, ask for weather on a date that does not exist, for example *weather in Paris on 2024-02-30*.
The reply says what was attempted and that it failed, and the tool step shows the error.

## What each reply shows

- **Tool steps**, above the answer: every tool call of the turn with its arguments and its result. Geocode, STAC and
  embedding results are drawn on a map.
- **The answer.** Values carry their source tool, for example `cloud cover 1.77% (search_stac_items, S2A_43QHV_...)`.
- **A verifier note** under the answer, when the verifier rewrote the reply or left a `⚠️ Could not verify…` caveat.
- **🧭 Graph trace.** The graph for that turn: `context → agent ⇄ tools → verify` (plus the TerraMind agent when it
  was called). The path that ran is in blue and numbered in execution order, with each node's run count and time.
  Hover a node for tokens in, turns in the prompt, whether a summary was included, how many earlier tool results were
  compacted, tools requested and the verifier's verdict.
- **Inspect a step.** Pick one node run to see what it was given, each model call (latency, tokens, response,
  requested tools), each tool call (arguments, result, duration, remote agent), and what it produced. The **Full
  prompt** popover shows exactly what the model was sent.

The graph trace comes from `GET /sessions/{id}/node_runs`.

![Graph trace of a TerraMind turn in the UI](../assets/images/graph_ui.png)

*One TerraMind turn. Blue numbers are the order the nodes ran: `context` (1), `agent` (2), then four `agent ⇄ tools` rounds (3-10) that made three calls to the TerraMind A2A agent (dashed green, 4.4 s of the tools' 6.8 s), then `verify` (11) and the answer (12).*



## Sidebar

| Control | What it does |
|---|---|
| **➕ New chat** | Opens a new tab with a fresh session id |
| **Load a session by ID** | Paste a session id and press **Load session**. The UI reads the stored conversation from the API (`GET /sessions/{id}`) and its trace, and opens it in a new tab. Works after an API restart, because sessions are in SQLite. Shows an error if that id is not in this API's database |
| **Open a session file** | Upload an exported session JSON and press **Replay file**. It opens read-only and needs no API history, so a JSON file can be sent to someone else who can read the whole test |
| **Try** | Example prompts to copy |

## Exporting a session

In a tab, press **Prepare session export**, then **⬇ Download session JSON**. The file holds the messages, the API
trace, the node-level traces, the TerraMind agent's log lines and the metadata of the embeddings it created
(`GET /sessions/{id}/export`). The same file comes from `python scripts/export_session.py <session_id>`.
`demo/session_terramind_example.json` is a ready-made one: upload it with **Open a session file**.

If you press **Prepare session export** before sending a message, the UI says there is nothing to export yet.

## Using the API without the UI

The same questions work from a terminal:

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"t1","message":"Weather in Paris on 2024-02-30"}' | jq '.reply, .trace[] | select(.error)'
```

or with the scripted demos, each of which prints a transcript and writes its trace:

| Script | Shows |
|---|---|
| `python scripts/run_demo.py` | the three required turns: a tool question, a context-dependent follow-up, a forced tool error |
| `python scripts/run_demo_context.py` | a 7-turn session where the rolling summary fires and turn 7 still recalls turn 1 |
| `python scripts/run_demo_terramind.py` | embedding and comparing two scenes through the A2A agent |

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Request failed" in the UI | the API is not running, or `EVE_API_BASE` points elsewhere |
| API does not start | the MCP server is down, or `GROQ_API_KEY` is missing |
| No embedding tools, "I can't embed scenes" | the TerraMind agent was not running when the API started: start it, then restart the API. Check `GET /health` |
| A TerraMind turn times out | first call loads the model and reads Sentinel-2 bands; allow a minute or more. The API waits up to `TERRAMIND_A2A_TIMEOUT` (180 s) per call |
| "No stored history for that session ID" | the id was typed wrongly, or the API uses a different `CHECKPOINT_DB_PATH` |
| `⚠️ Could not verify…` under an answer | the verifier found a value that is not in the tool results; open the graph trace and the `verify` node to see which |
