# Hallucination control: how it works, how well, and what it misses

Detail for the *Hallucination control* section of the [README](../README.md). The tests and the benchmark are in
[evals/README.md](../evals/README.md) and [evals/SCENARIOS.md](../evals/SCENARIOS.md).

## In one minute

Two layers, both about **evidence**: the prompt tells the model to use only tool results, and a `verify` node checks the
final answer against those results with plain rules. There is no second model call.

```
final answer ──► verify ──► every date, id, number, superlative and cited tool checks out? ── yes ──► reply
                              │ no
                              ▼
                 one rewrite (model sees this turn's tool results + the offending values)
                              │
                              ▼
                 checks again ── ok ──► reply
                              │ still failing
                              ▼
                 reply + visible "⚠️ Could not verify…" listing the values
```

The verify node (`agents/graphs/verify.py`) runs once per turn, after the tool loop. Each turn adds a `verify` step to the
trace with its verdict (`issues`, `rewritten`, `caveat`).

**Prompt rules.** Every fact must come from a tool result, errors are reported rather than hidden, and each value names its
source tool, for example `cloud cover 3% (search_stac_items, S2C_43PGQ_…)`. Without document retrieval, that is what
"citation" means here.

## The four checks

| Check | Question it answers | Caught example | Code |
|---|---|---|---|
| **Value** | Does this date, scene id or number exist in a tool result, the rolling summary or the user's own words? | An invented scene id; a cloud cover that no scene has | `grounding.py` |
| **Pairing** | Is the value on the right record: the right scene's cloud cover, the right day's temperature? | 1.16% attached to the scene that has 4.66%; Jan 26's wind put on the Jan 28 row | `pairing.py` |
| **Superlative** | Is "clearest / hottest / wettest / windiest / most similar" true? The min or max is recomputed from the tool data; ties pass | "Clearest is X (2.64%)" when another scene has 1.16% | `pairing.py` |
| **Tool citation** | Was the tool named as a source actually called in this session? | `31.2 (get_weather)` when `get_weather` never ran | `grounding.py` |

**What counts as "supported"** for a number, since a reply legitimately rounds and summarises: it appears in a tool result to the
precision the reply used (17.3601 supports "17.36"); it is a count, min, max, mean or sum of a list in a result ("7 scenes",
"max 31.2 °C"); it is a ratio shown as a percent (0.73 → 73%); the user said it; or it is a whole number quoted from a tool's own
description ("16 days ahead").

**Kept conservative on purpose**, because a false alarm on a correct answer costs more than a miss:
- pairing only judges a line or sentence that names **exactly one** record, and skips comparisons ("than", "vs", ranges) and
  hedges ("among the clearest", "second clearest");
- only decimal numbers are paired (small integers collide with counts and tile numbers);
- values in advice sentences ("e.g.", "try", "would you like") are exempt, scene ids never.

The superlative and attribution words are regex patterns (`SUPERLATIVES` in `pairing.py`); the verdict itself is `min()` / `max()`
over the parsed tool data. No model is involved.

## How well it works

Measured on a labelled benchmark (`python -m evals.eval_detection`): 284 replies, 247 hallucinated and 37 correct, in ten scenarios.
Overall **precision 100%, recall 89%, F1 94%**, and 0 false alarms on the 37 correct replies (upper bound 9%; the sample is small).

| Strong (100% recall) | Weak |
|---|---|
| a value on the wrong scene or the wrong day, invented scene ids, shifted dates, false superlatives, tool cited but never called | plain numbers in a dense payload (about 33% of random wrong 1-decimal values pass), derived values such as an average (40% caught), A2A numbers such as grid cells and days apart (56%), a false value echoed from the user (0%), wrong values hidden in advice text (0%) |

Per-scenario precision, recall and F1 are in the README's *Eval suite* section. The benchmark is synthetic and written by one
person, so it describes how the checker handles *these* error types, not hallucination in general.

Also measured on 137 final replies stored from this project's runs: 0 false alarms; the 2 it flagged were real errors, and it
caught 1,266 of 1,271 deliberately swapped values. The 25 live cases are scored on the same checks (`groundedness.py`), with
`no_rewrite: true` on cases where the verifier must not interfere.

## Scenarios not covered

| Not covered | Why it slips through | Status |
|---|---|---|
| **Claims with no value** ("mostly clear", "land change unlikely") | nothing numeric to check | needs an LLM judge or NLI |
| **Intent** (did it answer what the user meant?) | the verifier checks evidence, not meaning | needs an LLM judge |
| **Units** (a tool's 27 °C reported as "27 °F (get_weather)") | the number and the tool both match | checkable deterministically; not built |
| **Derived values** (averages, differences, conversions) | a wrong average can equal another data point; a right one outside count/min/max/mean/sum is flagged | partial; a labelled dataset is the next step |
| **Superlative phrasings that are not listed** | only the words in `SUPERLATIVES` are recomputed. Found in a real session: a table row "58.1% (lowest)" and "the best available scene" passed while an 18.0% scene existed | open; add the phrasings and a test |
| **A swap inside a sentence naming several records**, or a record referred to by position ("the second one") | pairing needs exactly one named record | by design |
| **Echoing a false value the user stated** | user numbers count as evidence | designed blind spot |
| **A wrong value inside advice** | advice sentences are exempt | designed blind spot |
| **A tool result that contains instructions** (a malicious MCP or A2A response) | tool output is trusted as data; only injection in the user's message is tested | not tested |
| **Rolling-summary faithfulness** | a summary error persists into every later turn and is checked nowhere | not built |
| **Embedding ids and small integers** | the id check covers uppercase scene ids; small whole numbers appear in many payloads | known limit |

The full map of scenarios, with the live case or test that covers each, is in [evals/SCENARIOS.md](../evals/SCENARIOS.md).

## Why not a second LLM call

The verifier is an **evidence check, not an intent check**. It is deterministic, costs no model call, and gives the same
verdict every time, which is why it can sit on every turn. A judge model would reach the rows above that are not covered
(claims without values, intent), but it is probabilistic and has to be calibrated before it can be trusted.

**Production path (not built):**
- A **model-based intent and alignment check** after the deterministic one, for sub-agents that used their own LLM to answer.
  Use a model from a different family than the one that wrote the answer. Candidates: a small LLM judge with a rubric, or an NLI
  model (weaker for intent); to be compared on labelled data.
- **Calibrate it against human-labelled data**: real replies from stored sessions, relabelled by a person, and report agreement with
  the labels. This also fixes the benchmark being written by the same person as the checker.
- **Unit-aware pairing**, derived-value checks and a check on tool-result injection, in the order given in `SCENARIOS.md` section 4.
- The unlisted superlative phrasings above, added to `SUPERLATIVES` with a test.

## Run the tests

```bash
python -m evals.test_groundedness   # the value check
python -m evals.test_pairing        # pairing and superlatives
python -m evals.test_verify         # the verify node: rewrite, caveat
python -m evals.eval_detection      # the 284-reply benchmark
python -m evals.run_evals --repeat 5   # the 25 live cases
```

The offline tests need no network or model.
