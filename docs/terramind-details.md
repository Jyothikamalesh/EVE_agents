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
