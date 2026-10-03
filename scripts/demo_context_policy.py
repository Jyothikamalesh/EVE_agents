"""Demonstrate the context-bounding policy actually triggering.

A 3-turn conversation (see run_demo.py) never gets close to the token
budget, so it can't show trimming kick in. This script builds a synthetic
long conversation with large tool payloads (the same shape real STAC/weather
results have) and runs it through the exact trimming call
``ReactAgent`` uses (``agents/graphs/react/graph.py::_invoke``), with a
deliberately small ``max_tokens`` so the effect is visible without needing
20+ real LLM round trips.

This exercises the mechanism in isolation — no live LLM or MCP calls needed.

Usage:
    source .venv/bin/activate
    python scripts/demo_context_policy.py
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages

from agents.graphs.utils import tiktoken_counter

SMALL_MAX_TOKENS = 800  # artificially low, to force visible trimming while still
# keeping at least one full prior turn — enough for "the second one" to resolve

_FAKE_STAC_PAYLOAD = str(
    {
        "catalog": "https://earth-search.aws.element84.com/v1",
        "items": [
            {
                "id": f"S2A_43QHV_2024{month:02d}{day:02d}_0_L2A",
                "datetime": f"2024-{month:02d}-{day:02d}T05:00:00Z",
                "cloud_cover": 3.2,
                "assets": ["blue", "green", "red", "nir", "visual", "thumbnail"],
            }
            for month, day in [(1, d) for d in range(1, 11)]
        ],
    }
)


def build_long_conversation(n_turns: int):
    messages = []
    for i in range(n_turns):
        messages.append(HumanMessage(content=f"Turn {i}: search imagery for city #{i}"))
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_stac_items",
                        "args": {"bbox": [0, 0, 1, 1]},
                        "id": f"call_{i}",
                    }
                ],
            )
        )
        messages.append(
            ToolMessage(
                content=_FAKE_STAC_PAYLOAD, tool_call_id=f"call_{i}", name="search_stac_items"
            )
        )
        messages.append(AIMessage(content=f"Found scenes for city #{i}."))
    # Final turn the demo question hangs off of — "the second one" needs turn-1 in context.
    messages.append(HumanMessage(content="Tell me more about the second one you found."))
    return messages


def main() -> None:
    conversation = build_long_conversation(n_turns=15)
    system = SystemMessage(content="You are an EO assistant.")
    full = [system] + conversation

    full_tokens = tiktoken_counter(full)
    print(f"Full conversation: {len(full)} messages, ~{full_tokens} tokens")
    print(f"Trimming to max_tokens={SMALL_MAX_TOKENS} (ReactAgent's policy, just a smaller budget)...")

    trimmed = trim_messages(
        full,
        max_tokens=SMALL_MAX_TOKENS,
        strategy="last",
        token_counter=tiktoken_counter,
        include_system=True,
        start_on="human",
        end_on=("human", "tool"),
    )

    trimmed_tokens = tiktoken_counter(trimmed)
    print(f"Trimmed conversation: {len(trimmed)} messages, ~{trimmed_tokens} tokens")

    dropped = len(full) - len(trimmed)
    print(f"\nDropped {dropped} oldest messages ({dropped / len(full):.0%} of the conversation).")
    print("System prompt kept:", isinstance(trimmed[0], SystemMessage))
    print("Final user question kept:", trimmed[-1].content)
    print("Most recent prior turn kept (so 'the second one' can resolve):")
    for m in trimmed:
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            print(" ", m.content)
    print(
        "\nThis is what 'bounded by a message window' means in practice: the raw tool "
        "payloads from turns 0-N get dropped once the budget is exceeded, while the "
        "system prompt and the most recent turns (which the live question depends on) "
        "are kept. Cross-session facts that must survive longer than the window belong "
        "in a summary (not implemented here — see README 'What I left out')."
    )


if __name__ == "__main__":
    main()
