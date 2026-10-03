"""Offline test: a model that keeps calling a failing tool is stopped after max_tool_calls.

Run: python -m evals.test_loop_guard
"""

import asyncio

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from agents.graphs.react.graph import ReactAgent


class StubbornLLM:
    """Always asks for the same tool call, never answers."""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content="", tool_calls=[{"name": "flaky", "args": {"x": "1"}, "id": f"c{self.calls}"}])


async def _flaky(x: str) -> str:
    raise RuntimeError("upstream is down")


def run(cap):
    llm = StubbornLLM()
    tool = StructuredTool.from_function(coroutine=_flaky, name="flaky", description="always fails")
    graph = ReactAgent().compile(llm=llm, tools=[tool], checkpointer=InMemorySaver(), max_tool_calls=cap, verify=True)
    result = asyncio.run(graph.ainvoke({"messages": [("user", "do it")]}, config={"configurable": {"thread_id": "t"}, "recursion_limit": 100}))
    return llm, result["messages"]


def test_loop_stops_at_cap_with_plain_explanation():
    llm, msgs = run(3)
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert llm.calls == 4                      # 3 executed calls + the 4th request that was refused
    assert sum("upstream is down" in str(m.content) for m in tool_msgs) == 3
    assert "Skipped" in str(tool_msgs[-1].content)  # the pending call is answered, so the history stays valid
    final = msgs[-1]
    assert isinstance(final, AIMessage) and not final.tool_calls and "tool-call limit" in final.content
    assert "upstream is down" in final.content   # the last real error is surfaced to the user


def test_no_cap_means_unchanged_behaviour():
    llm = StubbornLLM()
    tool = StructuredTool.from_function(coroutine=_flaky, name="flaky", description="always fails")
    graph = ReactAgent().compile(llm=llm, tools=[tool], checkpointer=InMemorySaver())
    try:
        asyncio.run(graph.ainvoke({"messages": [("user", "x")]}, config={"configurable": {"thread_id": "t"}, "recursion_limit": 8}))
    except Exception as exc:  # GraphRecursionError: what the cap exists to prevent
        assert "recursion" in type(exc).__name__.lower() or "recursion" in str(exc).lower()
    else:
        raise AssertionError("expected the uncapped loop to hit the recursion limit")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
