"""HTTP API for the Tier 1 EO agent (assignment section 2.5).

Run:
    source .venv/bin/activate
    python -m mcp_server.server &          # MCP server, separate process
    uvicorn service.api:app --port 8000    # this API

POST /chat {"session_id": "...", "message": "..."} -> {"session_id", "reply", "trace"}
GET  /health

Session id -> LangGraph checkpointer thread id, 1:1. Same session id across
calls keeps conversation state (context policy — see README).

Checkpointer is SQLite-backed (`AsyncSqliteSaver`), not in-memory: conversation
state survives an API restart, since the PoC stage is done — see README
"Context policy" for what this does and doesn't fix (single-box durability;
not a multi-host checkpointer, that's still AsyncPostgresSaver territory).
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from agents.graphs.verify import tool_capability_text
from service.eo_agent import EOReactAgent
from service.eo_agent.compaction import EO_COMPACTORS
from service.a2a_client import load_a2a_tools
from service.llm import describe as describe_llm
from service.llm import make_llm
from service.mcp_client import load_traced_tools
from service.node_trace import NodeTraceHandler, load_turns, save_turn
from service.sessions import export_session, serialize_messages
from service.tracing import record_step, start_trace

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(session_id)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

MODEL = describe_llm()  # "provider:model", see service/llm.py
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH", str(DATA_DIR / "checkpoints.sqlite"))
# Context policy (always on): tool results from earlier turns are compacted in the prompt, and turns
# older than the last KEEP_RECENT are folded into a rolling summary on demand, when the prompt for the
# new turn would exceed SUMMARY_TOKEN_BUDGET tokens (short chats never pay for a summary call).
# Set EVE_SUMMARY_TOKEN_BUDGET=0 to use a fixed schedule instead: every SUMMARY_EVERY aged-out turns.
SUMMARY_TOKEN_BUDGET = int(os.environ.get("EVE_SUMMARY_TOKEN_BUDGET", "3000"))
SUMMARY_EVERY = int(os.environ.get("EVE_SUMMARY_EVERY", "3"))
KEEP_RECENT = int(os.environ.get("EVE_KEEP_RECENT_TURNS", "2"))
# Loop guard: tool calls allowed per turn before the agent stops and explains.
MAX_TOOL_CALLS = int(os.environ.get("EVE_MAX_TOOL_CALLS", "10"))

_session_id_var: ContextVar[str] = ContextVar("_session_id_var", default="unknown")


class _SessionFilter(logging.Filter):
    """Stamp every log line with the session id of the request it belongs to (grep a session id)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = _session_id_var.get()
        return True


for _h in logging.getLogger().handlers:
    _h.addFilter(_SessionFilter())
_graph = None  # set during lifespan startup
_llm = None
_remote_tools: dict[str, str] = {}  # tool name -> remote agent label (A2A tools), for the node trace
_tool_info: list[dict] = []  # name, description and capability text of every bound tool, listed by /tools
_tool_names: list[str] = []  # every tool bound to the agent, listed by /health (the evals use it)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _llm, _remote_tools, _tool_names, _tool_info
    if os.environ.get("EVE_LLM_PROVIDER", "groq").lower() == "groq" and not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set (expected in .env), or choose another EVE_LLM_PROVIDER")

    Path(CHECKPOINT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as checkpointer:
        await checkpointer.setup()  # no-op after the first run; creates tables once
        # Tools: the EO MCP server (required) plus the TerraMind A2A agent's skills (optional).
        a2a_tools = await load_a2a_tools(session_id_getter=lambda: _session_id_var.get())
        _remote_tools = {t.name: "TerraMind A2A agent" for t in a2a_tools}
        tools = await load_traced_tools(lambda: _session_id_var.get(), a2a_tools)
        _tool_names = [t.name for t in tools]
        _tool_info = [{"name": t.name, "description": t.description, "text": tool_capability_text(t)} for t in tools]
        _llm = make_llm()
        _graph = EOReactAgent().compile(
            llm=_llm,
            tools=tools,
            checkpointer=checkpointer,
            tool_compactors=EO_COMPACTORS,
            summary_token_budget=SUMMARY_TOKEN_BUDGET or None,
            summary_every=None if SUMMARY_TOKEN_BUDGET else SUMMARY_EVERY,
            keep_recent_turns=KEEP_RECENT,
            verify=True,
            max_tool_calls=MAX_TOOL_CALLS,
        )
        logger.info(
            "Agent ready: model=%s tools=%s checkpoint_db=%s",
            MODEL,
            [t.name for t in tools],
            CHECKPOINT_DB_PATH,
        )
        yield


app = FastAPI(title="EVE Tier 1 EO Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    trace: list
    turn: int | None = None  # 1-based turn number; GET /sessions/{id}/node_runs has its node-level trace


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "tools": _tool_names}


@app.get("/tools")
async def tools():
    """What each bound tool says about itself (description + parameter schema); the verifier treats whole
    numbers in it as supported facts about the tool's limits, and the evals use the same evidence."""
    return {"tools": _tool_info}


async def _session_state(session_id: str) -> dict:
    values = (await _graph.aget_state({"configurable": {"thread_id": session_id}})).values or {}
    if not values.get("messages"):
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    return values


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """The stored conversation (tool calls and results included), for the UI and for reloading by id."""
    values = await _session_state(session_id)
    return {
        "session_id": session_id,
        "messages": serialize_messages(values["messages"]),
        "summary": values.get("summary") or None,
        "summarized_turns": values.get("summarized_turns") or 0,
    }


@app.get("/sessions/{session_id}/node_runs")
async def get_node_runs(session_id: str):
    """Node-level trace per turn: node runs in order, with model calls (full prompts) and tool calls."""
    return {"session_id": session_id, "turns": load_turns(session_id)}


@app.get("/sessions/{session_id}/export")
async def get_session_export(session_id: str):
    """One JSON file with the whole session: messages, API trace, TerraMind agent log, embeddings."""
    return export_session(session_id, await _session_state(session_id), MODEL)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    token = _session_id_var.set(req.session_id)
    steps = start_trace()
    start = time.perf_counter()
    try:
        node_trace = NodeTraceHandler(remote_tools=_remote_tools)
        config = {"configurable": {"thread_id": req.session_id}, "callbacks": [node_trace]}
        before = (await _graph.aget_state(config)).values or {}
        result = await _graph.ainvoke({"messages": [("user", req.message)]}, config=config)
        if result.get("summarized_turns", 0) != before.get("summarized_turns", 0):
            record_step(
                session_id=req.session_id,
                step="context_summary",
                args={"summarized_turns": result["summarized_turns"], "token_budget": SUMMARY_TOKEN_BUDGET or None,
                      "every": None if SUMMARY_TOKEN_BUDGET else SUMMARY_EVERY, "keep_recent": KEEP_RECENT},
                final_answer=result.get("summary"),
            )
        v = result.get("verification")
        if v is not None:
            record_step(
                session_id=req.session_id,
                step="verify",
                args={k: v[k] for k in ("issues", "rewritten", "caveat") if k in v},
                error=("; ".join(v.get("remaining") or v["issues"]) if v["issues"] else None) if v.get("caveat") else None,
            )
        turn = sum(type(m).__name__ == "HumanMessage" for m in result["messages"])
        save_turn(req.session_id, turn, node_trace.runs)
        final_message = result["messages"][-1]
        reply = final_message.content or "(no response generated)"
    finally:
        _session_id_var.reset(token)

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    record_step(
        session_id=req.session_id,
        step="agent_turn",
        duration_ms=duration_ms,
        final_answer=reply,
    )
    return ChatResponse(session_id=req.session_id, reply=reply, trace=steps, turn=turn)
