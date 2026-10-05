# Recorded runs

Reports from the runs behind the numbers in the README and `evals/SCENARIOS.md`. Each `<stamp>.md` is the
readable summary; `<stamp>.json` has every reply, tool call and verdict.

- `20261004-222556`: Baseline: all 25 cases x 5 repeats (125 runs, 125 passed)
- `20261004-230336`: Exploratory: 27 harder scenarios run once (17 passed, 10 failed; classified in evals/SCENARIOS.md)
- `20261004-232748`: tool_error_bad_date after the stricter assertion, 5 repeats (5/5)
- `detection-20261004-232511`: Detection benchmark: precision / recall / F1 per scenario and error type

**Traces.** Every request writes a step-by-step trace, `logs/node_runs/<session_id>.jsonl`: node, tool, arguments,
duration, error, the verifier's verdict and the final answer. The session ids of a run are in its `.json`
(`results[].session_id`), so any case can be matched to its trace. Only the traces of the runs above are kept here.
