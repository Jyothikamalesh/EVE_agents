# Evals

25 cases in `cases.yaml` across tool use, multi-turn context (including a 7-turn session that forces the
summary to fire), planning, refusal and prompt injection, robustness, the error path, hallucination bait
(including a false premise stated by the user), the A2A agent and the tools' own limits. Each case runs in a fresh session
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
python -m evals.run_evals --repeat 5                # every case 5 times: passes per case + pass rate with a 95% interval
python -m evals.run_evals --base http://127.0.0.1:18000 --db <checkpoints.sqlite>
python -m evals.test_groundedness               # offline tests of the checker
```

`--db` must point at the SQLite file the API writes to (`CHECKPOINT_DB_PATH`, default
`data/checkpoints.sqlite`). Reports go to `evals/results/<timestamp>.{json,md}`; exit code is 1
if any case fails (with `--min-pass-rate 0.9`, if the share of passing runs is below 90%).

## Repeat runs: how reliable is a case?

Model output is not deterministic even at temperature 0, so a single pass per case says little.
`--repeat N` runs every case N times, each in a fresh session, and prints passes per case
(`4/5`, lowest first), the overall pass rate with a 95% Wilson interval, and two lists: **flaky**
(passed some runs) and **failing** (never passed). The report keeps every run and counts the most
common failure messages per case.

All N runs passing is not "100%". The lower end of the 95% interval when every run passes:

| runs (N) | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|
| lower bound | 57% | 72% | 84% | 89% | 93% |

so "reliable above 90%" needs about 30 clean runs per case, and 5 runs is enough to find flaky
cases but not to certify a rate. Cases that skip (`requires_tools`) are not counted.

The report also shows how often the **verifier intervened** (flagged, rewrote, left a caveat), because the
cases are scored on the final reply: a pass that needed a rewrite is not the same as a model that was right
the first time.

**Measured** (`--repeat 3`, all 25 cases): **75/75 runs passed** (95% CI 95% to 100% pooled; each case
alone is 3/3, which supports only about 44%), 102/102 turns grounded, no flaky case. The verifier flagged
**4 of 102 turns (4%)**: 3 were a real, repeatable model error (in a 31-row weather table the model puts the Jan 26
and Jan 29 values on the Jan 28 row, and the pairing check catches and fixes it every time) and 1 was a derived value
(the model subtracted two similarity scores, which the strict check treats as unsupported by design).

**What the first `--repeat 5` run found, and the fix.** On the original 18 cases (90/90 passing) the verifier
had flagged and rewritten 20% of turns, always in `forecast_horizon`, `empty_result`, `tool_error_bad_date`
and `terramind_unknown_embedding_id`. The drafts were good: a capability statement quoting the weather tool's own
limits ("16 days ahead", "back to 1940"), suggested thresholds ("e.g. <=10%"), suggested valid dates, and tool names
mentioned in backticks. The strict check rewrote them into worse answers (the forecast reply became "no tools were
called"). Three narrow fixes took those four cases to **0 of 25** flagged turns: whole numbers quoted from a tool's
own description count as supported (decimals still need a result), values in advisory sentences ("e.g.", "try",
"would you like") are exempt unless they also appear in a plain claim (scene ids never are), and a tool name only
counts as a citation when attached to a value (`31.2 (get_weather)`). `no_rewrite: true` in `cases.yaml` now makes
a case fail if the verifier had to rewrite the reply, and the runner fetches `GET /tools` so the evals judge
groundedness with the same evidence as the live verifier.

**Coverage, honestly.** Added after an audit (17 of the first 19 cases were single-turn, `rank_similar` was never
exercised and nothing tested false premises, injection, table grounding or the TerraMind reading guide):
`false_premise_cloud_cover`, `weather_table_ten_days`, `long_session_recall`, `prompt_injection_user`,
`terramind_rank_then_no_percent_sameness`, `future_date_imagery` (plus `capability_question`). Still not covered:
ambiguous place names (a known gap: geocoding returns one match by default), the loop guard live, tool-result
prompt injection, non-English input, latency or cost, and concurrent sessions.

## Detection benchmark and the scenario map

Every `run_evals` run also prints and records the benchmark's headline precision / recall / F1 (offline, no model; `--skip-detection` turns it off). `python -m evals.eval_detection` gives the full per-scenario tables and scores the verifier as a hallucination detector: 284 labelled replies
(`detection_fixtures.py`) in ten scenarios, precision / recall / F1 per scenario and per error type, a
false-alarm rate, and a sweep of how often a random wrong number passes. `test_detection.py` pins the
results, including the known weaknesses. **`SCENARIOS.md`** explains why each scenario was chosen, maps
every other scenario (covered, partial or not), records 27 exploratory scenarios tried once, and lists where
a labelled true/false dataset would be better than a live case.

## Offline tests (no network, no model)

```bash
python -m evals.test_groundedness      # the checker
python -m evals.test_pairing           # record pairing + superlatives (right value, right scene/day)
python -m evals.test_verify            # the verifier node (rewrite / caveat)
python -m evals.test_context_policy    # compaction + rolling summary
python -m evals.test_terramind_skills  # embed / compare / rank logic, no-data guard
python -m evals.test_loop_guard        # tool-call cap stops a retry loop
python -m evals.test_text_tool_calls   # <tool_use> / <function=> / [TOOL_CALLS] parsing
python -m evals.test_llm_factory       # provider selection (groq/ollama/vllm/openrouter/hf)
python -m evals.test_node_trace        # node-level trace recorder (UI graph data)
python -m evals.test_a2a_client        # A2A tools described by the Agent Card; the EO prompt names no tools
python -m evals.test_run_evals         # repeat statistics: Wilson interval, flaky/failing detection, verifier interventions, report
python -m evals.test_detection         # the detection benchmark's results, incl. the known weaknesses
```

The `a2a` cases in `cases.yaml` need `python -m terramind_agent.server` running. A case can list `requires_tools`;
the runner reads the bound tools from `GET /health` and reports the case as `SKIP`, not `FAIL`, when one is missing.
