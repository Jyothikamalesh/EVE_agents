"""Offline tests for the verifier node. Run: python -m evals.test_verify"""

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.graphs.verify import make_verify_node

TOOLS = ["geocode_location", "search_stac_items", "get_weather"]
SEARCH = json.dumps({"count": 1, "items": [{"id": "S2B_31UDQ_20240105_0_L2A", "datetime": "2024-01-05T10:50:21Z", "cloud_cover": 18.01}]})


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    async def ainvoke(self, _msgs):
        self.calls += 1
        return AIMessage(content=self.reply)


def state(reply, with_tool=True):
    msgs = [HumanMessage(content="imagery over Paris?")]
    if with_tool:
        msgs += [AIMessage(content="", tool_calls=[{"name": "search_stac_items", "args": {}, "id": "1"}]),
                 ToolMessage(content=SEARCH, tool_call_id="1", name="search_stac_items")]
    msgs.append(AIMessage(content=reply, id="final"))
    return {"messages": msgs}


def run(reply, llm_reply="", with_tool=True):
    llm = FakeLLM(llm_reply)
    out = asyncio.run(make_verify_node(llm, TOOLS)(state(reply, with_tool)))
    return out, llm


def test_grounded_passes_untouched():
    out, llm = run("Scene S2B_31UDQ_20240105_0_L2A on 2024-01-05, cloud 18% (search_stac_items).")
    assert "messages" not in out and out["verification"]["issues"] == [] and llm.calls == 0


def test_ungrounded_is_rewritten():
    out, llm = run("Scene S2B_31UDQ_20240199_0_L2A has 4% cloud.", "Scene S2B_31UDQ_20240105_0_L2A has 18% cloud (search_stac_items).")
    assert llm.calls == 1 and out["verification"]["rewritten"] and not out["verification"]["caveat"]
    assert out["messages"][0].id == "final" and "20240105" in out["messages"][0].content


def test_unfixable_gets_caveat():
    out, _ = run("Cloud was 77%.", "Cloud was 77%.")
    assert out["verification"]["caveat"] and "Could not verify" in out["messages"][0].content


def test_uncalled_tool_citation_flagged():
    out, _ = run("Max 31.2 (get_weather).", "Max 31.2 (get_weather).")
    assert any("get_weather" in i for i in out["verification"]["issues"])


def test_small_ints_ignored_without_tools():
    out, llm = run("I have 3 tools: geocoding, STAC search and weather.", with_tool=False)
    assert out["verification"]["issues"] == [] and llm.calls == 0


def test_memory_number_without_tools_flagged():
    out, llm = run("Kilimanjaro is 5895 m tall.", "I can't verify that.", with_tool=False)
    assert llm.calls == 1 and out["verification"]["rewritten"]


# ── real drafts the verifier used to rewrite into worse answers (from a 90-run eval) ──────────────────
WEATHER_DESCRIPTION = (
    "Get daily weather for a point: historical, recent, or forecast. Pass coordinates (use geocode_location first "
    "if the user named a place). With no dates, returns today plus a 7-day forecast. Historical requests route "
    "automatically to the long-range archive (back to 1940); recent/future requests route to the forecast API "
    "(up to 16 days ahead)."
)
FORECAST_DRAFT = (
    "I can't answer that one. The weather tool I have (get_weather) only reaches about 16 days into the future, "
    "so a forecast for Paris six months from now is outside its range.\n\nWhat I *can* do:\n"
    "- Give you the **7-day forecast** for Paris (today + next 7 days)\n"
    "- Pull **historical daily weather** for Paris from a specific past period (back to 1940) \u2014 e.g. what it was "
    "actually like in Paris around this time last year\n\nWant either of those?"
)
EMPTY_DRAFT = (
    "No Sentinel-2 L2A scenes over Paris in January 2024 had exactly 0% cloud cover (search_stac_items, "
    "bbox [2.224122, 48.8155755, 2.4697602, 48.902156], 2024-01-01 to 2024-01-31, max_cloud_cover=0 \u2192 0 results).\n\n"
    "January is a cloudy month in Paris, so this is expected. Would you like me to relax the filter "
    "(e.g. \u226410% or \u226420% cloud cover) to find the clearest available scenes?"
)
BAD_DATE_DRAFT = (
    "I called `get_weather` exactly as you asked \u2014 with `start_date='2024-02-30'` and `end_date='2024-02-30'`.\n\n"
    "The tool failed with this error:\n\n> **Error executing tool get_weather: day is out of range for month**\n\n"
    "That's because February 30th doesn't exist. If you'd like, I can re-run it for a valid date (e.g. 2024-02-29, "
    "which was a real day since 2024 was a leap year, or 2024-03-01). Just let me know which."
)
EMBED_DRAFT = (
    "I can't compare those \u2014 `emb_does_not_exist` and `emb_also_missing` are not embedding IDs that exist in this "
    "conversation.\n\nIf you'd like a real comparison, I can:\n1. Search for Sentinel-2 scenes over a place you name,\n"
    "2. Embed one or two of them with `embed_scene`,\n3. Then run `compare_embeddings` on the resulting IDs."
)


def run_with(reply, outputs=None, user="question", tool_texts=None, tools=TOOLS + ["embed_scene", "compare_embeddings"]):
    """Run the node on a turn whose tool results are *outputs* ({tool: content}); nothing else is shown to it."""
    msgs = [HumanMessage(content=user)]
    for i, (name, content) in enumerate((outputs or {}).items()):
        msgs += [AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": str(i)}]),
                 ToolMessage(content=content, tool_call_id=str(i), name=name)]
    msgs.append(AIMessage(content=reply, id="final"))
    llm = FakeLLM("REWRITTEN")
    out = asyncio.run(make_verify_node(llm, tools, tool_texts)({"messages": msgs}))
    return out, llm


def test_capability_statement_is_not_rewritten():
    out, llm = run_with(FORECAST_DRAFT, tool_texts=[WEATHER_DESCRIPTION], user="Will it rain in Paris in six months?")
    assert out["verification"]["issues"] == [] and llm.calls == 0
    # the numbers are only supported because the tool's own description says so
    out, _ = run_with(FORECAST_DRAFT, user="Will it rain in Paris in six months?")
    assert any("number 16" in i or "number 1940" in i for i in out["verification"]["issues"])


def test_suggested_thresholds_and_dates_are_advice_not_claims():
    geo = json.dumps({"best_match": {"display_name": "Paris", "bbox": [2.224122, 48.8155755, 2.4697602, 48.902156]}})
    out, llm = run_with(EMPTY_DRAFT, {"geocode_location": geo, "search_stac_items": json.dumps({"count": 0, "date_range": "2024-01-01/2024-01-31", "items": []})},
                        user="Sentinel-2 over Paris in January 2024 with exactly 0% cloud")
    assert out["verification"]["issues"] == [] and llm.calls == 0, out["verification"]["issues"]
    out, llm = run_with(BAD_DATE_DRAFT, {"get_weather": "Error executing tool get_weather: day is out of range for month"},
                        user="call get_weather with start_date='2024-02-30' and end_date='2024-02-30'")
    assert out["verification"]["issues"] == [] and llm.calls == 0, out["verification"]["issues"]


def test_a_tool_that_is_only_mentioned_is_not_a_citation():
    out, llm = run_with(EMBED_DRAFT, user="Compare emb_does_not_exist and emb_also_missing")
    assert out["verification"]["issues"] == [] and llm.calls == 0, out["verification"]["issues"]
    out, _ = run_with("Similarity is 0.998 (compare_embeddings).", user="compare them")        # attached to a value: a citation
    assert any("compare_embeddings" in i for i in out["verification"]["issues"])


def test_the_rewrite_prompt_keeps_explanations_and_suggestions():
    from agents.graphs.verify import REWRITE_SYSTEM
    assert "explanations of what you cannot do" in REWRITE_SYSTEM and "suggestions" in REWRITE_SYSTEM


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("ok ", f.__name__)
    print(f"{len(fns)} passed")
