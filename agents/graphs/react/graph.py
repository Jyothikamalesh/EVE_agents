"""ReAct agent graph — manual tool-calling loop via LangGraph StateGraph.

Uses shared utilities from the parent ``graphs`` package (``utils``) for
text-format tool-call parsing and message sanitisation.  Imports are
relative so this tree can be cloned as its own repository.
"""

import logging
from typing import Any, Callable, Dict, List, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from ..base import AgentGraph, AgentMessagesState
from ..context import (
    DEFAULT_MAX_CHARS,
    Compactor,
    compact_old_tool_messages,
    summarize_turns,
    ledger_block,
    request_ledger,
    summary_block,
    turn_starts,
)
from ..verify import make_verify_node, tool_capability_text
from ..policies import (
    DEFAULT_LLM_IDLE_TIMEOUT,
    DEFAULT_LLM_RUN_TIMEOUT,
    LLM_RETRY,
    build_llm_fallback_timeout_policy,
    llm_node_add_kwargs,
    make_llm_fallback_error_handler,
)
from ..utils import (
    parse_text_tool_calls,
    reformat_messages_for_text_tool_model,
    strip_content_from_tool_call_messages,
    tiktoken_counter,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 96_000


class ReactAgent(AgentGraph):
    """Manual ReAct loop: agent -> tools -> agent, with text-format fallback.

    Supports models with native function calling (OpenAI-style) and models
    that emit tool calls as text (Mistral/EVE-Instruct ``[TOOL_CALLS]`` format).

    Pass ``fallback_llm`` from the backend to enable in-graph model fallback.
    The primary ``agent`` node retries transient failures in-place (``LLM_RETRY``)
    and, once retries are exhausted, an ``error_handler`` routes to a dedicated
    ``agent_fallback`` node that runs the fallback model.  Tool failures are
    surfaced back to the agent as ``ToolMessage`` content so the ReAct loop can
    recover, rather than retried at the node level (a node-level retry would
    re-invoke every tool call in the turn).
    """

    name = "react"

    def compile(
        self,
        *,
        llm,
        tools: List[BaseTool],
        checkpointer: Any,
        history: Optional[List[Any]] = None,
        summary: Optional[str] = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        fallback_llm=None,
        llm_run_timeout: Optional[float] = DEFAULT_LLM_RUN_TIMEOUT,
        llm_idle_timeout: Optional[float] = DEFAULT_LLM_IDLE_TIMEOUT,
        on_policy=None,
        extra_instruction: Optional[Callable[[], Optional[str]]] = None,
        tool_compactors: Optional[Dict[str, Compactor]] = None,
        summary_every: Optional[int] = None,
        summary_token_budget: Optional[int] = None,
        summary_min_turns: int = 2,
        keep_recent_turns: int = 2,
        compact_max_chars: int = DEFAULT_MAX_CHARS,
        verify: bool = False,
        max_tool_calls: Optional[int] = None,
        **kwargs,
    ):
        # Context policy (both opt-in; see graphs/context.py):
        # - ``tool_compactors`` (a dict, possibly empty): tool results from earlier turns are
        #   compacted in the prompt (per-tool function, else a length cap). Stored history is untouched.
        # - ``summary_token_budget=T`` (on-demand): when the prompt for the new turn would exceed T
        #   tokens, fold every turn older than the last ``keep_recent_turns`` into a rolling summary
        #   (one LLM call, at least ``summary_min_turns`` turns at a time) and drop them from the
        #   prompt. Short conversations never pay for it.
        # - ``summary_every=N`` (fixed schedule, used when no budget is given): fold once N turns
        #   have aged out of the last ``keep_recent_turns`` verbatim turns.
        summary_on = bool(summary_every or summary_token_budget)
        base_instruction = self.instruction_text(history=history, summary=summary)
        primary_llm_bound = llm.bind_tools(tools) if tools else llm
        fallback_llm_bound = (
            fallback_llm.bind_tools(tools)
            if (fallback_llm is not None and tools)
            else fallback_llm
        )
        has_fallback = fallback_llm_bound is not None

        # what the model would be sent for this state, before the hard token trim
        def _prompt_messages(state: AgentMessagesState):
            messages = list(state["messages"])
            summary = state.get("summary") or ""
            ledger = None
            if summary_on:
                # Turns already folded into the summary leave the prompt (they stay in state).
                # The user's own words for those turns stay as a numbered ledger, in order.
                starts = turn_starts(messages)
                done = int(state.get("summarized_turns") or 0)
                if 0 < done < len(starts):
                    ledger = ledger_block(request_ledger(messages, done))
                    messages = messages[starts[done] :]
            if tool_compactors is not None:
                messages = compact_old_tool_messages(messages, tool_compactors, compact_max_chars)
            extra = extra_instruction() if extra_instruction else None
            parts = (base_instruction, summary_block(summary), ledger, extra)
            instruction = "\n\n".join(p for p in parts if p) or None
            if instruction:
                messages = [SystemMessage(content=instruction)] + messages
            return messages

        # shared invocation logic
        async def _invoke(state: AgentMessagesState, llm_bound):
            messages = _prompt_messages(state)

            if trim_messages is not None:
                messages = trim_messages(
                    messages,
                    max_tokens=max_tokens,
                    strategy="last",
                    token_counter=tiktoken_counter,
                    include_system=True,
                    start_on="human",
                    end_on=("human", "tool"),
                )

            messages = strip_content_from_tool_call_messages(messages)

            has_synthetic = any(
                (
                    isinstance(m, AIMessage)
                    and not m.content
                    and getattr(m, "tool_calls", None)
                )
                or isinstance(m, ToolMessage)
                for m in messages
            )
            if has_synthetic:
                messages = reformat_messages_for_text_tool_model(messages)

            response = await llm_bound.ainvoke(messages)

            if not getattr(response, "tool_calls", None) and isinstance(
                response.content, str
            ):
                parsed = parse_text_tool_calls(response.content)
                if parsed:
                    logger.info(
                        "Parsed %d text-format tool call(s) from model response",
                        len(parsed),
                    )
                    response = AIMessage(
                        content="",
                        tool_calls=parsed,
                        id=getattr(response, "id", None),
                    )

            return {"messages": [response]}

        # rolling-summary node: runs once per request, before the agent
        async def context_fn(state: AgentMessagesState):
            messages = state["messages"]
            starts = turn_starts(messages)
            done = int(state.get("summarized_turns") or 0)
            # turns old enough to leave the verbatim window (the current turn is the last one)
            target = len(starts) - 1 - keep_recent_turns
            if target <= done:
                return {}
            if summary_token_budget:
                # on demand: only when the prompt this turn would be sent is over budget
                size = tiktoken_counter(_prompt_messages(state))
                if size <= summary_token_budget:
                    return {}
                # hysteresis: batch at least ``summary_min_turns`` turns per call (so a tight budget
                # cannot cost one summary call per turn), unless the prompt is far over budget
                if target - done < summary_min_turns and size <= 1.5 * summary_token_budget:
                    return {}
                logger.info("Prompt is %d tokens (budget %d): folding turns %d..%d", size, summary_token_budget, done + 1, target)
            elif target - done < summary_every:
                return {}
            chunk = messages[starts[done] : starts[target]]
            try:
                new_summary = await summarize_turns(
                    llm, state.get("summary") or "", chunk, tool_compactors, compact_max_chars
                )
            except Exception:  # noqa: BLE001 - keep the old summary; the token trim is the backstop
                logger.exception("Rolling summary failed; keeping previous summary")
                return {}
            if not new_summary:
                return {}
            logger.info("Rolling summary updated: turns %d..%d folded in", done + 1, target)
            return {"summary": new_summary, "summarized_turns": target}

        # primary agent node
        async def agent_fn(state: AgentMessagesState):
            return await _invoke(state, primary_llm_bound)

        # fallback agent node (no further error_handler - failures bubble)
        async def agent_fallback_fn(state: AgentMessagesState):
            return await _invoke(state, fallback_llm_bound)

        # ── routing ────────────────────────────────────────────────────────
        # with ``verify`` the final answer passes through the verifier node before ending
        after_agent = "verify" if verify else END

        def _turn_tool_calls(messages) -> int:
            starts = turn_starts(messages)
            return sum(isinstance(m, ToolMessage) for m in messages[starts[-1]:]) if starts else 0

        def should_continue(state: AgentMessagesState) -> Literal["tools", "limit", "verify", "__end__"]:
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None):
                if max_tool_calls and _turn_tool_calls(state["messages"]) >= max_tool_calls:
                    return "limit"
                return "tools"
            return after_agent

        # Loop guard: a model that keeps retrying a failing tool would otherwise run until the
        # recursion limit. Answer the pending calls as skipped and end the turn with a plain message.
        async def limit_fn(state: AgentMessagesState):
            last = state["messages"][-1]
            msgs = state["messages"]
            starts = turn_starts(msgs)
            errors = [str(m.content) for m in msgs[starts[-1]:] if isinstance(m, ToolMessage) and "error" in str(m.content).lower()]
            skipped = [
                ToolMessage(content="Skipped: tool-call limit reached for this turn.", tool_call_id=tc["id"], name=tc["name"])
                for tc in last.tool_calls
            ]
            detail = f" The last error was: {errors[-1][:300]}" if errors else ""
            text = (
                f"I stopped because this turn hit the tool-call limit without reaching an answer.{detail} "
                "Please rephrase or narrow the request and I'll try again."
            )
            return {"messages": skipped + [AIMessage(content=text)]}

        # ── build graph ────────────────────────────────────────────────────
        builder = StateGraph(AgentMessagesState)
        builder.add_node(
            "agent",
            self.timed_node("agent", agent_fn, on_policy=on_policy),
            **llm_node_add_kwargs(
                fallback_node="agent_fallback",
                has_fallback=has_fallback,
                llm_run_timeout=llm_run_timeout,
                llm_idle_timeout=llm_idle_timeout,
                on_policy=on_policy,
            ),
        )
        builder.add_node("tools", self.make_tools_node(tools))
        route_map = {"tools": "tools", END: END}
        if max_tool_calls:
            builder.add_node("limit", limit_fn)
            builder.add_edge("limit", "verify" if verify else END)
            route_map["limit"] = "limit"
        if verify:
            builder.add_node("verify", make_verify_node(llm, [t.name for t in tools], [tool_capability_text(t) for t in tools]))
            builder.add_edge("verify", END)
            route_map["verify"] = "verify"
        if summary_on:
            builder.add_node("context", context_fn)
            builder.add_edge(START, "context")
            builder.add_edge("context", "agent")
        else:
            builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue, route_map)
        builder.add_edge("tools", "agent")

        if has_fallback:
            fallback_node_kwargs: dict[str, Any] = {
                "retry_policy": LLM_RETRY,
                "error_handler": make_llm_fallback_error_handler(
                    fallback_node="agent_fallback",
                    has_fallback=has_fallback,
                    on_policy=on_policy,
                ),
            }
            timeout = build_llm_fallback_timeout_policy(
                run_timeout=llm_run_timeout, idle_timeout=llm_idle_timeout
            )
            if timeout is not None:
                fallback_node_kwargs["timeout"] = timeout
            builder.add_node(
                "agent_fallback",
                self.timed_node(
                    "agent_fallback", agent_fallback_fn, on_policy=on_policy
                ),
                **fallback_node_kwargs,
            )
            builder.add_conditional_edges("agent_fallback", should_continue, route_map)

        return builder.compile(checkpointer=checkpointer)
