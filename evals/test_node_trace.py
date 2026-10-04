"""Offline test: the callback handler records node runs, model calls and tool calls.

Run: python -m evals.test_node_trace
"""

import asyncio
import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from agents.graphs.react.graph import ReactAgent
from service import node_trace
from service.node_trace import NodeTraceHandler


class ScriptedLLM(BaseChatModel):
    """A real LangChain chat model (so callbacks fire): first call asks for a tool, then answers."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.calls += 1
        if "running summary" in str(messages[0].content):
            msg = AIMessage(content="SUMMARY")
        elif self.calls == 1:
            msg = AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"q": "rome"}, "id": "c1"}])
        else:
            msg = AIMessage(content="Rome is at 41.9 (lookup).")
        return ChatResult(generations=[ChatGeneration(message=msg)])


async def _lookup(q: str) -> str:
    return json.dumps({"lat": 41.9, "q": q})


def run(handler, **compile_kw):
    tool = StructuredTool.from_function(coroutine=_lookup, name="lookup", description="d")
    graph = ReactAgent().compile(llm=ScriptedLLM(), tools=[tool], checkpointer=InMemorySaver(), **compile_kw)
    return asyncio.run(graph.ainvoke({"messages": [("user", "where is rome?")]},
                                     config={"configurable": {"thread_id": "t"}, "callbacks": [handler]}))


def test_records_node_sequence_with_children():
    h = NodeTraceHandler(remote_tools={"lookup": "Remote agent"})
    run(h, summary_every=3, verify=True)
    assert [r["node"] for r in h.runs] == ["context", "agent", "tools", "agent", "verify"]
    assert [r["seq"] for r in h.runs] == [1, 2, 3, 4, 5]
    assert all(r["status"] == "ok" and r["duration_ms"] is not None for r in h.runs)

    agent1, tools, agent2 = h.runs[1], h.runs[2], h.runs[3]
    assert len(agent1["llm_calls"]) == 1
    call = agent1["llm_calls"][0]
    assert call["prompt_summary"]["turns_in_prompt"] == 1 and call["prompt_summary"]["messages"] >= 1
    assert any(m["role"] == "user" and "rome" in m["content"] for m in call["prompt"])    # full prompt kept
    assert call["response"]["tool_calls"] == [{"name": "lookup", "args": {"q": "rome"}}]
    assert agent1["output"]["requested_tools"] == ["lookup"]

    (tc,) = tools["tool_calls"]
    assert tc["name"] == "lookup" and tc["args"] == {"q": "rome"} and json.loads(tc["result"])["lat"] == 41.9
    assert tc["remote"] == "Remote agent" and tc["error"] is None and tc["duration_ms"] is not None

    assert agent2["output"]["final_answer"].startswith("Rome is at")
    assert agent2["llm_calls"][0]["prompt_summary"]["tool_results"] == 1
    assert h.runs[4]["output"]["verification"]["issues"] == []


def test_tool_error_is_recorded():
    async def boom(q: str) -> str:
        raise RuntimeError("upstream down")

    h = NodeTraceHandler()
    tool = StructuredTool.from_function(coroutine=boom, name="lookup", description="d")
    graph = ReactAgent().compile(llm=ScriptedLLM(), tools=[tool], checkpointer=InMemorySaver())
    asyncio.run(graph.ainvoke({"messages": [("user", "x")]}, config={"configurable": {"thread_id": "t"}, "callbacks": [h]}))
    tools = next(r for r in h.runs if r["node"] == "tools")
    assert "upstream down" in (tools["tool_calls"][0]["error"] or "")


def test_persistence_roundtrip(tmp_path=None):
    import tempfile, pathlib
    node_trace.NODE_RUNS_DIR = pathlib.Path(tempfile.mkdtemp())
    node_trace.save_turn("s/1", 1, [{"seq": 1, "node": "agent"}])
    node_trace.save_turn("s/1", 2, [{"seq": 1, "node": "agent"}])
    node_trace.save_turn("s/1", 2, [{"seq": 1, "node": "agent"}, {"seq": 2, "node": "tools"}])  # re-run wins
    turns = node_trace.load_turns("s/1")
    assert [t["turn"] for t in turns] == [1, 2] and len(turns[1]["node_runs"]) == 2
    assert node_trace.load_turns("nope") == []


def test_prompt_is_capped():
    from langchain_core.messages import HumanMessage, ToolMessage
    big = [HumanMessage(content="q"), ToolMessage(content="x" * 500_000, tool_call_id="1", name="t")]
    out = node_trace.serialize_prompt(big)
    assert sum(len(d["content"]) for d in out) < 200_000 and "truncated" in out[1]["content"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
