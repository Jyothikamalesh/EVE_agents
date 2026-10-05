# Context policy: how it works, how well, and what is not built

Detail for the *Context policy* section of the [README](../README.md).

## In one minute

The full conversation is **stored**. What the **model sees** each turn is a smaller view built from it. The two are kept
separate, so shrinking the prompt never loses anything.

```
 stored (SQLite, by session id)                      sent to the model each turn
 ───────────────────────────────                     ───────────────────────────────────────────
 every message, tool call and        context node    system prompt
 raw tool result, nothing deleted  ───────────────►  + rolling summary of folded turns   (layer 3)
                                                     + numbered list of the user's own
                                                       requests for those turns          (layer 4)
                                                     + last turns, verbatim, with earlier
                                                       tool output shrunk                (layer 2)
```

Short chats pay for none of the extra work: the summary only appears once the prompt would pass the token budget.

## The four layers

### 1. Stored per session
- The full LangGraph message list, keyed by `thread_id = session_id` in an `AsyncSqliteSaver` (`data/checkpoints.sqlite`).
- Nothing is dropped from storage. It survives an API restart (verified by killing `uvicorn` and asking a follow-up in a fresh process).
- Single-box durability. `AsyncPostgresSaver` is the same `thread_id` interface for several hosts.

### 2. Tool-output compaction
Code: `agents/graphs/context.py`, with the per-tool rules in `service/eo_agent/compaction.py`.

When the prompt is built, tool results from **earlier turns** are replaced by a compact form and tagged
`[compacted; call the tool again for full detail]`. The current turn is never compacted, and the stored history is untouched (the UI,
the evals and the verifier still see the raw payloads).

| Tool | What is kept |
|---|---|
| `search_stac_items` | count, collections, date range, bbox, and per scene: id, date, cloud cover |
| `get_stac_item` | id, collection, datetime, cloud cover, bbox, the asset *names* |
| `get_weather` | place, dates, and per variable min and max with their dates, and the mean (a total for sums); short series are kept whole |
| `geocode_location` | name, lat, lon, bbox of the best match and up to two others |
| `list_stac_collections` | collection ids |
| any other tool | output cut at 1,200 characters, with a `[truncated N chars; …]` note |

If a compactor fails the code falls back to the 1,200-character cap, so compaction can never break a turn.
Example: a `search_stac_items` result of 5,720 characters becomes 662.

### 3. Rolling summary, on demand
- When the prompt for the new turn would exceed `EVE_SUMMARY_TOKEN_BUDGET` (default 3,000 tokens), every turn older than the last
  `EVE_KEEP_RECENT_TURNS` (default 2) is folded into a running summary by **one LLM call**, and those turns leave the prompt.
- The summary keeps facts only: places, bbox, dates, filters, scene ids, errors.
- At least 2 turns are folded per call, so a tight budget cannot cost one call per turn (unless the prompt is 1.5× over budget).
- A failed summary call keeps the old summary. A 96k-token `trim_messages` window sits behind everything as the last backstop.
- `EVE_SUMMARY_TOKEN_BUDGET=0` switches to a fixed schedule instead (every `EVE_SUMMARY_EVERY` aged-out turns, default 3).
- The 3,000-token default was calibrated against today's prompt: re-check it if the prompt changes.

### 4. Request ledger
A summary keeps facts but loses order, so "the very first thing I asked" went wrong: the model answered with the latest request.
For every turn folded into the summary, the prompt also carries the user's own words, numbered and in order
(`request_ledger` in `agents/graphs/context.py`).
- Built in code from the stored messages, with **no LLM**, so it cannot drift or invent.
- Costs about 30 tokens per turn.


## Design decisions

### Current
- **One ReAct loop** (agent node, tools node, `verify` node on the final answer) with a context step that builds the prompt each turn from the four layers above.
- **Eight tools:** five over MCP, three from the TerraMind A2A agent (described by its Agent Card). Tool calls per turn are capped (`EVE_MAX_TOOL_CALLS`).
- **Checkpointer:** SQLite for development and single-box use; Postgres for production.
- **Serving:** the API calls the graph, which binds the tools the MCP server and the Agent Card advertise at start-up.

### Why not the alternatives
- **Supervisor with sub-agents.** With 8 tools, choosing a tool is not the bottleneck, and a supervisor adds a model call and a lossy hand-off
  per turn. Reconsider at roughly 20+ tools, or when domains need different prompts, permissions or models.
- **ReWOO (plan everything first).** It plans before it has seen any tool result, so a failed call or a call that depends on an earlier result
  (geocode, then search, then embed) leaves the rest of the plan wrong and needs a replanning step. That is a ReAct loop with extra machinery.
  A plan step with replanning is the middle option if tasks become long and multi-stage.
- **ReAct** fits because each step's input is the previous step's output and errors arrive as observations the model can react to.
- **Qdrant / RAG inside a session.** Within one session, even a long one, the summary plus the request ledger keeps order and exact wording,
  which similarity retrieval does not (it returns what is similar, not what was asked first). RAG is the right tool across sessions.

## Known limits
- The rolling summary is **not checked for faithfulness**: a summary error persists into every later turn (see `evals/SCENARIOS.md` section 4).
- The ledger keeps the user's words but **not the assistant's earlier conclusions**.
- The 3,000-token budget is calibrated to the current prompt.

## Production path (not built)
Each item names the trigger that would justify it.
- **Planner node and supervisor** once several MCP servers, agents and sub-systems are attached: a planner at the start, sub-agents behind it, and
  a verifier that can call a model (see [hallucination-details.md](hallucination-details.md)) so a sub-system that answers with its own LLM is
  checked for intent against the user's request, not only for values.
- **Cross-session memory with RAG** (what "memory" means in a chat app), also useful if EVE exposes an MCP server. Session memories in
  **Qdrant**, with a separate general-purpose document retrieval system alongside. Settle first: Qdrant vs `pgvector`, since Postgres is already
  the production checkpointer (Qdrant is justified by filtered search at scale or dedicated vector operations, otherwise `pgvector` is one less
  service). Memory also needs a write policy, staleness handling and privacy rules.
- **A faithfulness check on the rolling summary,** and a way to carry the assistant's conclusions across folded turns.


## Settings

| Variable | Default | Effect |
|---|---|---|
| `EVE_SUMMARY_TOKEN_BUDGET` | 3000 | fold older turns once the prompt would pass this (0 = fixed schedule) |
| `EVE_KEEP_RECENT_TURNS` | 2 | turns kept verbatim |
| `EVE_SUMMARY_EVERY` | 3 | only used when the budget is 0 |
| `EVE_MAX_TOOL_CALLS` | 10 | tool calls allowed per turn (the loop guard; set the same way, but not part of the context policy) |

## Demonstrated and measured

**Demonstrated:** [`demo/demo_transcript_context.md`](../demo/demo_transcript_context.md) and `demo/demo_trace_context.jsonl`
(`python scripts/run_demo_context.py`), a 7-turn session.
- The prompt grows from about 450 to 3,100 tokens over turns 1-5.
- At the start of turn 6 it would pass the 3,000-token budget, so turns 1-3 are folded (a `context_summary` trace step) and the prompt drops to about 2,500.
- Turn 7 asks about turn 1 ("which city, what cloud limit, what dates?"), which has left the prompt, and is answered correctly.
- For scale, the raw stored history of that session is about 9,700 tokens (a tiktoken count of the stored messages; the prompt figures are the node trace's approximate count).

**Measured:** the same conversation run 5 times answered turn 7 correctly **5 of 5** with the ledger (and again 5 of 5 after switching to the
on-demand trigger). Before the ledger it was **1 of 3**: the summary held the Hyderabad request, but the model reported the most recent one
(Nairobi, 30%) as the first. The verifier cannot catch that failure, since every value in the wrong answer exists in the tool data.
`evals/test_context_policy.py` covers the mechanics offline.

## How it fits with the verifier

Compaction is applied only when the prompt is built. The stored messages stay raw and the verifier reads the stored tool messages
(`verify.py`), not the compacted prompt, so a compacted earlier-turn value cannot cause a false alarm. The rolling summary is also
accepted as evidence.


## Run it

```bash
python scripts/run_demo_context.py      # the 7-turn session; writes demo/demo_transcript_context.md
python -m evals.test_context_policy     # compaction, summary trigger and ledger, offline
```
