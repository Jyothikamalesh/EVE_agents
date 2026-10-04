"""Offline tests for compaction + rolling summary (fake LLM, no network).

Run: python -m evals.test_context_policy
"""

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agents.graphs.context import compact_old_tool_messages, ledger_block, request_ledger, turn_starts
from agents.graphs.react.graph import ReactAgent
from agents.graphs.utils import tiktoken_counter
from service.eo_agent.compaction import EO_COMPACTORS

SEARCH = {
    "catalog": "x", "collections": ["sentinel-2-l2a"], "bbox": [1, 2, 3, 4], "date_range": "2024-01-01/2024-01-31",
    "count": 2,
    "items": [
        {"id": f"S2B_{i}_20240105_0_L2A", "collection": "sentinel-2-l2a", "datetime": "2024-01-05T10:50:21Z",
         "cloud_cover": 18.0104, "bbox": [1, 2, 3, 4], "assets": ["visual", "B04", "B08"] * 10,
         "thumbnail": "https://example.com/" + "x" * 200}
        for i in range(2)
    ],
}
WEATHER = {"source": "archive", "lat": 1.0, "lon": 2.0, "start_date": "2024-01-01", "end_date": "2024-01-10",
           "daily": {"time": [f"2024-01-{d:02d}" for d in range(1, 11)],
                     "temperature_2m_max": [20.0 + d for d in range(10)], "precipitation_sum": [0.0] * 9 + [5.5]},
           "daily_units": {"temperature_2m_max": "°C"}}


class FakeLLM:
    """Records every prompt; answers 'ok', or 'SUMMARY n' when asked to summarise."""

    def __init__(self):
        self.prompts, self.summaries = [], 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        if "running summary" in str(messages[0].content):
            self.summaries += 1
            return AIMessage(content=f"SUMMARY {self.summaries}")
        self.prompts.append(messages)
        return AIMessage(content="ok")


def _turn(n):
    return [HumanMessage(content=f"q{n}", id=f"h{n}"),
            AIMessage(content="", tool_calls=[{"name": "search_stac_items", "args": {}, "id": f"c{n}"}], id=f"a{n}"),
            ToolMessage(content=str(SEARCH), tool_call_id=f"c{n}", name="search_stac_items", id=f"t{n}"),
            AIMessage(content=f"a{n}", id=f"r{n}")]


def test_old_tool_messages_compacted_current_turn_raw():
    msgs = _turn(1) + _turn(2)
    out = compact_old_tool_messages(msgs, EO_COMPACTORS)
    old, cur = out[2], out[6]
    assert len(old.content) < len(str(SEARCH)) / 3 and "compacted" in old.content
    assert "S2B_0_20240105_0_L2A" in old.content and "2024-01-05" in old.content  # ids/dates survive
    assert "https://example.com" not in old.content and "B04" not in old.content   # bulk dropped
    assert cur.content == str(SEARCH)                                              # current turn untouched
    assert msgs[2].content == str(SEARCH)                                          # input not mutated


def test_weather_compaction_keeps_extremes_not_arrays():
    from agents.graphs.context import compact_content
    out = compact_content("get_weather", str(WEATHER), EO_COMPACTORS)
    assert "max_date" in out and "2024-01-10" in out and "29.0" in out and "daily_units" not in out
    assert len(out) < len(str(WEATHER))


def test_errors_and_unknown_tools_pass_through_or_cap():
    from agents.graphs.context import compact_content
    assert compact_content("get_weather", "Tool error: bad date", EO_COMPACTORS) == "Tool error: bad date"
    long = compact_content("mystery", "x" * 5000, EO_COMPACTORS, max_chars=100)
    assert long.startswith("x" * 100) and "truncated 4900" in long


def test_summary_schedule_and_prompt_contents():
    llm = FakeLLM()
    graph = ReactAgent().compile(llm=llm, tools=[], checkpointer=InMemorySaver(),
                                 tool_compactors=EO_COMPACTORS, summary_every=3, keep_recent_turns=2)
    cfg = {"configurable": {"thread_id": "t"}}
    covered = []

    async def run():
        for n in range(1, 10):
            r = await graph.ainvoke({"messages": [("user", f"turn {n}")]}, config=cfg)
            covered.append(r.get("summarized_turns", 0))
        return r

    final = asyncio.run(run())
    # first fold at turn 6 (turns 1-3), next at turn 9 (turns 4-6): one LLM call per 3 turns
    assert covered == [0, 0, 0, 0, 0, 3, 3, 3, 6], covered
    assert llm.summaries == 2

    humans = lambda p: [m.content for m in p if isinstance(m, HumanMessage)]
    p5, p6, p9 = llm.prompts[4], llm.prompts[5], llm.prompts[8]
    assert humans(p5) == [f"turn {n}" for n in range(1, 6)]            # nothing summarised yet
    assert humans(p6) == ["turn 4", "turn 5", "turn 6"]                # turns 1-3 left the prompt
    assert "SUMMARY 1" in p6[0].content                                # ...and are stood in for by the summary
    assert humans(p9) == ["turn 7", "turn 8", "turn 9"] and "SUMMARY 2" in p9[0].content
    # raw history is never destroyed: all 9 turns remain in the checkpoint
    assert len(turn_starts(final["messages"])) == 9


def test_request_ledger_is_ordered_verbatim_and_bounded():
    msgs = [HumanMessage(content="Imagery of Hyderabad, Jan 2024, <10% cloud?"), AIMessage(content="ok"),
            HumanMessage(content="  Weather\nthere?  "), AIMessage(content="ok"),
            HumanMessage(content="x" * 500), HumanMessage(content="turn 4")]
    lines = request_ledger(msgs, upto_turns=3)
    assert lines[0] == "1. Imagery of Hyderabad, Jan 2024, <10% cloud?"   # the user's own words, first
    assert lines[1] == "2. Weather there?"                                 # whitespace collapsed
    assert lines[2].startswith("3. xxx") and lines[2].endswith("\u2026") and len(lines[2]) < 260
    assert len(lines) == 3                                                 # only the turns asked for
    assert request_ledger(msgs, 0) == [] and ledger_block([]) is None


def test_ledger_reaches_the_prompt_only_for_folded_turns():
    llm = FakeLLM()
    graph = ReactAgent().compile(llm=llm, tools=[], checkpointer=InMemorySaver(),
                                 tool_compactors=EO_COMPACTORS, summary_every=3, keep_recent_turns=2)
    cfg = {"configurable": {"thread_id": "t"}}

    async def run():
        for n in range(1, 8):
            await graph.ainvoke({"messages": [("user", f"question number {n}")]}, config=cfg)

    asyncio.run(run())
    before, after = llm.prompts[4][0].content, llm.prompts[5][0].content   # turn 5 vs turn 6 (summary fires)
    assert "What the user asked in earlier turns" not in before
    assert "1. question number 1\n2. question number 2\n3. question number 3" in after
    assert "4. question number 4" not in after                              # turn 4 is still verbatim in the prompt
    assert "not from the summary" in after


def _run_budget(budget, turns, words=60, **kw):
    llm = FakeLLM()
    graph = ReactAgent().compile(llm=llm, tools=[], checkpointer=InMemorySaver(), tool_compactors=EO_COMPACTORS,
                                 summary_token_budget=budget, keep_recent_turns=2, **kw)
    cfg = {"configurable": {"thread_id": "t"}}
    folded = []

    async def run():
        for n in range(1, turns + 1):
            r = await graph.ainvoke({"messages": [("user", f"turn {n} " + "word " * words)]}, config=cfg)
            folded.append(r.get("summarized_turns", 0))
        return r

    return llm, folded, asyncio.run(run())


def test_token_budget_short_conversation_never_summarises():
    llm, folded, _ = _run_budget(budget=100_000, turns=10)
    assert llm.summaries == 0 and folded == [0] * 10


def test_token_budget_summarises_only_when_the_prompt_is_over_budget():
    probe, _, _ = _run_budget(budget=100_000, turns=1)
    base = tiktoken_counter(probe.prompts[0])                       # system prompt + the first turn
    llm, folded, final = _run_budget(budget=base + 250, turns=8)    # each further turn adds ~75 tokens
    first = next(i for i, f in enumerate(folded) if f)            # 0-based index of the first fold
    assert 2 <= first <= 5, folded                                  # not at once, but well before the end
    # everything older than the 2 verbatim turns is folded in one call, so the prompt shrinks again
    assert folded[first] == first - 2, folded
    # hysteresis: at least 2 turns are folded per call, so a tight budget is not one summary per turn
    assert llm.summaries <= 3 and len(set(folded)) <= 4, (llm.summaries, folded)
    humans = lambda p: sum(isinstance(m, HumanMessage) for m in p)
    assert humans(llm.prompts[first - 1]) == first                 # nothing folded yet: every turn verbatim
    assert humans(llm.prompts[first]) == 3                          # then: the current turn + 2 recent ones
    assert "SUMMARY 1" in llm.prompts[first][0].content
    assert len(turn_starts(final["messages"])) == 8               # stored history untouched


def test_token_budget_takes_precedence_over_the_fixed_schedule():
    llm, folded, _ = _run_budget(budget=100_000, turns=8, summary_every=3)
    assert llm.summaries == 0 and folded == [0] * 8


def test_disabled_by_default():
    llm = FakeLLM()
    graph = ReactAgent().compile(llm=llm, tools=[], checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}

    async def run():
        for n in range(1, 8):
            await graph.ainvoke({"messages": [("user", f"turn {n}")]}, config=cfg)

    asyncio.run(run())
    assert llm.summaries == 0 and len([m for m in llm.prompts[-1] if isinstance(m, HumanMessage)]) == 7


def test_summary_failure_keeps_going():
    class Boom(FakeLLM):
        async def ainvoke(self, messages):
            if "running summary" in str(messages[0].content):
                raise RuntimeError("llm down")
            return await super().ainvoke(messages)

    llm = Boom()
    graph = ReactAgent().compile(llm=llm, tools=[], checkpointer=InMemorySaver(), summary_every=3, keep_recent_turns=2)
    cfg = {"configurable": {"thread_id": "t"}}

    async def run():
        for n in range(1, 8):
            r = await graph.ainvoke({"messages": [("user", f"turn {n}")]}, config=cfg)
        return r

    r = asyncio.run(run())
    assert not r.get("summary") and r["messages"][-1].content == "ok"   # replies still produced, history intact


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
