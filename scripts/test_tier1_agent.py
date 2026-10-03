"""Manual end-to-end check: ReactAgent + TIER1_TOOLS against a live Groq model.

Not a pytest suite — a throwaway script to confirm the LLM actually chains
geocode -> STAC/weather tool calls correctly. Requires GROQ_API_KEY (see .env).

Usage:
    source .venv/bin/activate
    python3 scripts/test_tier1_agent.py
"""

import asyncio
import os

from dotenv import load_dotenv
from service.llm import make_llm
from langgraph.checkpoint.memory import InMemorySaver

from agents.graphs.react.graph import ReactAgent
from agents.tools import TIER1_TOOLS

load_dotenv()

MODEL = "qwen/qwen3.8-27b"

PROMPTS = [
    "Show me cloud-free Sentinel-2 imagery of Hyderabad from January 2024.",
    "What's the historical weather in Paris for the first week of July 2023?",
]


async def run_prompt(graph, prompt: str, thread_id: str) -> None:
    print(f"\n{'=' * 80}\nUSER: {prompt}\n{'=' * 80}")
    result = await graph.ainvoke(
        {"messages": [("user", prompt)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    for msg in result["messages"]:
        kind = type(msg).__name__
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"[{kind}] tool_call -> {tc['name']}({tc['args']})")
        elif kind == "ToolMessage":
            content = str(msg.content)
            print(f"[{kind}:{msg.name}] {content[:300]}")
        else:
            print(f"[{kind}] {msg.content}")


async def main() -> None:
    llm = make_llm()
    checkpointer = InMemorySaver()
    graph = ReactAgent().compile(llm=llm, tools=TIER1_TOOLS, checkpointer=checkpointer)

    for i, prompt in enumerate(PROMPTS):
        await run_prompt(graph, prompt, thread_id=f"test-{i}")


if __name__ == "__main__":
    asyncio.run(main())
