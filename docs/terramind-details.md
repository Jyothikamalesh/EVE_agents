# TerraMind A2A agent: full detail

Detail for the *TerraMind agent over A2A* section of the [README](../README.md).

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
20 m). Demo: `python scripts/run_demo_terramind.py` writes `../demo/demo_transcript_terramind.md` and
the matching trace. This replaced an earlier TerraMind MCP server, so there is now a single MCP server.


## A2A setup: running it, the card, the wire format

The TerraMind agent is a separate service, not a node in the EO agent's graph. It speaks A2A 1.0 (official `a2a-sdk`,
JSON-RPC, non-streaming). The API side of the link is in [API_MCP_A2A_interaction.md](API_MCP_A2A_interaction.md); this section is the
agent's own side.

**Run it.**

```bash
python -m terramind_agent.server        # http://127.0.0.1:8767  (also started by scripts/start_all.sh)
```

| Variable | Default | Meaning |
|---|---|---|
| `TERRAMIND_A2A_HOST`, `TERRAMIND_A2A_PORT` | `127.0.0.1`, `8767` | where it listens |
| `TERRAMIND_A2A_URL` | `http://<host>:<port>` | the public URL written into the Agent Card; also what the API uses to find it |
| `TERRAMIND_LOG_PATH` | `logs/terramind.jsonl` | one JSON line per skill call |
| `TERRAMIND_STORE_DIR` | `data/embeddings/` | saved embeddings |
| `TERRAMIND_A2A_TIMEOUT` | `180` (API side) | seconds the API waits for one call |

Start it **before** the API: the API reads the card once at start-up. If the agent is not reachable then, the API logs
a warning and runs without the three TerraMind tools; start the agent later and the tools do not appear until the
API is restarted.

**The Agent Card.** Published at `GET /.well-known/agent-card.json` (the real card, descriptions shortened here):

```json
{
  "name": "TerraMind EO embedding agent",
  "description": "Embeds Sentinel-2 scenes with the TerraMind foundation model and compares them.",
  "version": "1.0.0",
  "supportedInterfaces": [{"url": "http://127.0.0.1:8767", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
  "capabilities": {"streaming": false, "pushNotifications": false},
  "defaultInputModes": ["application/json"],
  "defaultOutputModes": ["application/json"],
  "skills": [
    {"id": "embed_scene", "name": "Embed a Sentinel-2 scene", "description": "Run a Sentinel-2 L2A scene through the TerraMind foundation model ...", "tags": ["earth-observation", "foundation-model", "embedding"]},
    {"id": "compare_embeddings", "name": "Compare two embedded scenes", "description": "Compare two scenes already embedded with embed_scene ...", "tags": ["earth-observation", "change-detection", "similarity"]},
    {"id": "rank_similar", "name": "Rank scenes by similarity", "description": "Rank already-embedded scenes by similarity to a reference scene ...", "tags": ["earth-observation", "similarity", "retrieval"]}
  ]
}
```

The skill `description` is written for the model: when to use the skill, how to read its numbers, what not to claim.
The EO agent shows it to the LLM unchanged as the tool description, so this agent owns its own usage instructions.
The card cannot declare typed parameters, so the argument schemas live in the client (`service/a2a_client.py`).

**One call, on the wire.** The client sends one message with one data part, and the agent answers with a task.

```text
request   message.parts[0].data = {"skill": "embed_scene",
                                   "args": {"collection": "sentinel-2-l2a", "item_id": "S2B_43QHV_20240103_0_L2A", "patch_size": 224}}
          message.context_id    = "<chat session id>"
          message.metadata      = {"session_id": "<chat session id>"}

task      SUBMITTED  →  WORKING  →  COMPLETED
artifact  name "embed_scene", one data part = the skill's JSON result:
          {"embedding_id": "emb_S2B_43QHV_20240103_0_L2A", "item_id": "S2B_43QHV_20240103_0_L2A",
           "datetime": "2024-01-03T05:24:04.428000Z", "cloud_cover": 0.866846, "tile_id": "43QHV",
           "crop_bbox": [78.33187, 17.5543, 78.35332, 17.57487], "patch_size": 224, "grid": [14, 14],
           "embedding_shape": [196, 192], "nodata_fraction": 0,
           "note": "The full tensor is stored server-side; pass embedding_id to compare_embeddings / rank_similar.", ...}
```

(From `demo/session_terramind_example.json`. The tensor itself never leaves the agent.)

**When a skill fails**, the task ends as `FAILED` with a text message and the server keeps running. A real example,
`compare_embeddings` on an id that was never embedded:

```text
task TASK_STATE_FAILED: unknown embedding_id 'emb_nope'; known ids: ['emb_S2A_43QHV_20240128_0_L2A', ...]
```

The EO agent raises it as `TerraMind agent: task TASK_STATE_FAILED: ...`, the tools node turns it into a tool error,
and the model explains it (the `terramind_unknown_embedding_id` eval case). A request that is not one data part with a
`skill` key also fails the task, with `send one data part: {"skill": <name>, "args": {...}}`.

**Session link and logs.** The chat session id arrives as `context_id` and `metadata.session_id`. Every call writes one
line to `logs/terramind.jsonl`, and each embedding is saved as `data/embeddings/<embedding_id>.npz` plus a `.json`
that records the creating session:

```json
{"timestamp": 1791092100.46, "session_id": "graph-demo-1", "task_id": "5cd16d24-...", "skill": "embed_scene",
 "args": {"collection": "sentinel-2-l2a", "patch_size": 224, "item_id": "S2B_43QHV_20240103_0_L2A"},
 "ok": true, "embedding_id": "emb_S2B_43QHV_20240103_0_L2A", "duration_ms": 36573.0}
```

`GET /sessions/{id}/export` on the API collects these lines and the embedding metadata for one session.

**Why a separate agent.** It owns a heavy model (about 1 GB of weights and `terratorch`), blocking inference and its
own storage, and it can be restarted, replaced or reused by another client without touching the EO agent. Its skills
are deterministic today, so it is a thin agent; making its executor LLM-driven would not change the card or the wire
format. Skills run in a worker thread (`asyncio.to_thread`) so the server stays responsive during inference.

**Check it by hand.**

```bash
curl -s localhost:8767/.well-known/agent-card.json | jq '.skills[].id'
python scripts/run_demo_terramind.py        # embeds, ranks and compares through the EO agent; writes demo/demo_transcript_terramind.md
python -m evals.test_a2a_client             # offline: tools are built from the card, the prompt names no tools
```


## How the number is produced, and the production path

### Current

`embed_scene` reads **six Sentinel-2 bands of one scene** over a 224-pixel crop, runs TerraMind, and keeps a
(196 tokens x 192 dimensions) tensor (14 x 14 grid of 16 px patches). `compare_embeddings` mean-pools the tokens to one
**192-dimensional vector** per scene and returns the **cosine similarity** between two scenes, plus a per-tile
similarity. The agent only handles ids and summary numbers. The scores say how alike two scenes *look*; they carry no
land-cover labels, and the reference points above are from three scenes, so they are indicative, not calibrated.

### Production path (not built)

- **Calibrate the score.** Collect human-labelled scene pairs (same / different, and what kind of difference) and fit
  thresholds from an ROC / equal-error-rate curve, per region or land-cover type, instead of the hand-measured
  reference points.
- **"It could be these areas."** A 192-d vector cannot say *what* a scene is. Keep human-labelled vectors in a vector
  store (the "real vector store is the next step" above) and answer by nearest neighbours: "closest to these labelled
  areas". Calibrate and evaluate that against held-out labels.
- **Check the interpretation.** The verifier already checks the numbers quoted from this agent; the wording rules
  (reading guide) are a candidate for a labelled dataset (`../evals/SCENARIOS.md`, section 4).
