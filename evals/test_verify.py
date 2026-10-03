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


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("ok ", f.__name__)
    print(f"{len(fns)} passed")
