# Context policy: demonstration and measurements

Detail for the *Context policy* section of the [README](../README.md).

**Demonstrated:** `../demo/demo_transcript_context.md` / `../demo/demo_trace_context.jsonl`
(`python scripts/run_demo_context.py`) — a 7-turn session. The prompt grows from about 450 to 3,100
tokens over turns 1-5; at the start of turn 6 it would pass the 3,000-token budget, so turns 1-3 are
folded (`context_summary` trace step) and the prompt drops to about 2,500 tokens. Turn 7 then asks
about turn 1 ("which city, what cloud limit, what dates?"), which has left the prompt, and is answered
correctly. For scale, the raw stored history of that session is about 9,700 tokens (tiktoken count of
the stored messages; the prompt figures are the node trace's approximate count).

**Measured:** the same 7-turn conversation run 5 times answered turn 7 correctly **5 of 5** with the
ledger (and again 5 of 5 after switching to the on-demand trigger). Before the ledger it was **1 of 3**: the summary held the Hyderabad request, but the model
reported the most recent one (Nairobi, 30%) as the first. The verifier cannot catch that failure,
since every value in the wrong answer exists in the tool data. `../evals/test_context_policy.py`
covers the mechanics offline.


## Original long-form description of the four layers

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
