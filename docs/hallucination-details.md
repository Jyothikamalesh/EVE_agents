# Hallucination control and evals: full detail

Detail for the *Hallucination control* section of the [README](../README.md).

- **Prompt rules:** every factual claim must come from a tool result; tool
  errors are reported, not papered over; and each value is cited with its
  source tool, e.g. `cloud cover 3% (search_stac_items, S2C_43PGQ_…)`. Without
  document retrieval, that provenance is what "citation" means here.
- **`verify` node** (`../agents/graphs/verify.py`, `../agents/graphs/grounding.py`):
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
  Beyond existence, `../agents/graphs/pairing.py` checks that a value sits with the
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
  advice text are missed. See `../evals/SCENARIOS.md`.
  Known limits: a swap inside a sentence that compares several records, a record referred to
  only by position ("the second one"), and claims with no value in them are not checked; it
  is a tripwire, not a proof.
- **Eval set** (`../evals/`, 25 cases: tool use, multi-turn context (incl. a 7-turn session), refusal, prompt injection,
  robustness, error path, hallucination): `python -m evals.run_evals`
  scores tool use, reply content and groundedness (`--repeat N` runs each case N times and reports passes per case, flaky cases and a pass rate with a 95% interval), and writes
  `../evals/results/<stamp>.{json,md}`. Offline tests:
  `python -m evals.test_groundedness`, `test_pairing`, `test_verify`, `test_context_policy`,
  `test_terramind_skills`, `test_loop_guard`, `test_text_tool_calls`, `test_llm_factory`,
  `test_node_trace`, `test_a2a_client`, `test_run_evals`, `test_detection`.
  See `../evals/README.md`.
