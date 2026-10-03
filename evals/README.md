# Evals

15 cases in `cases.yaml` across tool use, multi-turn context, planning, refusal,
robustness, the error path and hallucination bait. Each case runs in a fresh session
against the live API (`POST /chat`) and is scored on:

- **Tool use** — expected / forbidden tools, minimum call counts, argument regexes,
  and (for the error case) a tool error recorded in the trace. From the API's inline trace.
- **Reply content** — regexes the reply must / must not match.
- **Groundedness** — deterministic, no LLM (`groundedness.py`): dates, scene IDs and
  numbers in the reply must trace to the tool outputs the model saw (read back from the
  session's checkpoint), or to the user's own words. Handles rounding, counts, min/max/mean
  and day-of-month derived from ISO dates. Limits: a wrong number that happens to equal
  another number in the payload passes; a correct value derived any other way
  (difference, ratio) is flagged. Turns that offer example thresholds use `grounded: ids_dates`.

```bash
# MCP server + API running (see main README), then:
python -m evals.run_evals                       # all cases
python -m evals.run_evals --only tool_error_bad_date
python -m evals.run_evals --base http://127.0.0.1:18000 --db <checkpoints.sqlite>
python -m evals.test_groundedness               # offline tests of the checker
```

`--db` must point at the SQLite file the API writes to (`CHECKPOINT_DB_PATH`, default
`data/checkpoints.sqlite`). Reports go to `evals/results/<timestamp>.{json,md}`; exit code is 1
if any case fails. Model output is not deterministic even at temperature 0, so re-run a failing
case a few times before treating it as a regression.

## Offline tests (no network, no model)

```bash
python -m evals.test_groundedness      # the checker
python -m evals.test_verify            # the verifier node (rewrite / caveat)
python -m evals.test_context_policy    # compaction + rolling summary
python -m evals.test_terramind_skills  # embed / compare / rank logic, no-data guard
python -m evals.test_loop_guard        # tool-call cap stops a retry loop
python -m evals.test_text_tool_calls   # <tool_use> / <function=> / [TOOL_CALLS] parsing
python -m evals.test_llm_factory       # provider selection (groq/ollama/vllm/openrouter/hf)
```

The `a2a` cases in `cases.yaml` need `python -m terramind_agent.server` running.
