"""TerraMind A2A server (Agent2Agent protocol, official ``a2a-sdk``).

Publishes an Agent Card at ``/.well-known/agent-card.json`` and a JSON-RPC endpoint at ``/``.
Each request is one skill call, sent as a message with a single data part:

    {"skill": "embed_scene", "args": {"collection": "sentinel-2-l2a", "item_id": "S2B_..."}}

and answered with a completed task whose artifact holds the skill's JSON result (or a failed
task with an error message). The skills are deterministic, so this is a thin agent; swapping
the executor for an LLM-driven one would not change the card or the wire format.

Run:  python -m terramind_agent.server      # http://127.0.0.1:8767
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import uvicorn
from a2a.helpers.proto_helpers import get_data_parts, new_data_artifact, new_task, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue_v2 import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from starlette.applications import Starlette

from .skills import SkillError, run_skill

logger = logging.getLogger("terramind_agent")

HOST = os.environ.get("TERRAMIND_A2A_HOST", "127.0.0.1")
PORT = int(os.environ.get("TERRAMIND_A2A_PORT", "8767"))
PUBLIC_URL = os.environ.get("TERRAMIND_A2A_URL", f"http://{HOST}:{PORT}")

LOG_PATH = Path(os.environ.get("TERRAMIND_LOG_PATH", str(Path(__file__).resolve().parent.parent / "logs" / "terramind.jsonl")))


def log_call(**entry) -> None:
    """One JSON line per skill call, keyed by session_id (grep/jq the log by session)."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": time.time(), **entry}, default=str) + "\n")


def session_of(context: RequestContext) -> str:
    """The EO agent's chat session id: message metadata, else the A2A context id."""
    meta = dict(context.message.metadata) if context.message and context.message.HasField("metadata") else {}
    return str(meta.get("session_id") or context.context_id or "unknown")


SKILL_CARDS = [
    AgentSkill(
        id="embed_scene",
        name="Embed a Sentinel-2 scene",
        description=(
            "Run a Sentinel-2 L2A scene through the TerraMind foundation model (a centred crop, 6 bands) and "
            "return an embedding_id plus the scene's date, cloud cover, tile id and basic statistics. The "
            "(196, 192) tensor stays server-side, so pass only embedding_ids around. Use it only when the user "
            "asks to embed, analyse, compare or find similar imagery at a representation level, after a "
            "scene search has given a collection and item_id; not for ordinary 'show me imagery' requests. "
            "The embedded crop is a 2.2 km square at the centre of the Sentinel-2 tile (about 110 km across), "
            "not necessarily at the place searched for: say so, using crop_bbox. Takes about 30 seconds per scene."
        ),
        tags=["earth-observation", "foundation-model", "embedding"],
        examples=['{"skill": "embed_scene", "args": {"collection": "sentinel-2-l2a", "item_id": "S2B_43QHV_20240113_0_L2A"}}'],
    ),
    AgentSkill(
        id="compare_embeddings",
        name="Compare two embedded scenes",
        description=(
            "Compare two scenes already embedded with embed_scene (needs their embedding_ids). Returns the "
            "overall cosine similarity and, when both scenes share a tile id, where they differ: per-tile "
            "mean/min/max and the 3 least-similar grid cells (row, col). If same_footprint is false, the "
            "per-tile comparison does not apply: say so. The result carries a reading_guide, reference_points "
            "and caveats: follow them when explaining it. Scores are compressed near 1.0, so explain results "
            "by ranking and by where tiles differ, never as a percentage of 'sameness', and do not name a "
            "type of change (the embeddings carry no land-cover labels)."
        ),
        tags=["earth-observation", "change-detection", "similarity"],
    ),
    AgentSkill(
        id="rank_similar",
        name="Rank scenes by similarity",
        description=(
            "Rank already-embedded scenes by similarity to a reference scene (needs embedding_ids from "
            "embed_scene). Each row carries days_apart and caveats; the result carries a reading_guide and "
            "reference_points: follow them. Treat near-equal scores (same third decimal) as tied."
        ),
        tags=["earth-observation", "similarity", "retrieval"],
    ),
]


def build_card() -> AgentCard:
    return AgentCard(
        name="TerraMind EO embedding agent",
        description="Embeds Sentinel-2 scenes with the TerraMind foundation model and compares them.",
        version="1.0.0",
        supported_interfaces=[AgentInterface(url=PUBLIC_URL, protocol_binding="JSONRPC", protocol_version="1.0")],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=SKILL_CARDS,
    )


class TerraMindExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is None:  # the protocol wants the Task before any status update
            await event_queue.enqueue_event(new_task(
                context.task_id, context.context_id, TaskState.TASK_STATE_SUBMITTED,
                history=[context.message] if context.message else None,
            ))
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.update_status(TaskState.TASK_STATE_WORKING)
        session_id, started, skill, args = session_of(context), time.perf_counter(), None, None
        try:
            parts = get_data_parts(context.message.parts) if context.message else []
            request = parts[0] if parts and isinstance(parts[0], dict) else None
            if request is None or "skill" not in request:
                raise SkillError('send one data part: {"skill": <name>, "args": {...}}')
            skill, args = request["skill"], request.get("args") or {}
            # model inference and COG reads are blocking: keep the event loop free
            result = await asyncio.to_thread(run_skill, skill, args, session_id)
        except SkillError as exc:
            logger.warning("[%s] skill rejected: %s", session_id, exc)
            log_call(session_id=session_id, task_id=context.task_id, skill=skill, args=args, ok=False,
                     error=str(exc), duration_ms=round((time.perf_counter() - started) * 1000, 1))
            await updater.update_status(TaskState.TASK_STATE_FAILED, new_text_message(str(exc)))
            return
        except Exception as exc:  # noqa: BLE001 - report to the caller, never crash the server
            logger.exception("[%s] skill failed", session_id)
            log_call(session_id=session_id, task_id=context.task_id, skill=skill, args=args, ok=False,
                     error=f"{type(exc).__name__}: {exc}", duration_ms=round((time.perf_counter() - started) * 1000, 1))
            await updater.update_status(TaskState.TASK_STATE_FAILED, new_text_message(f"{type(exc).__name__}: {exc}"))
            return
        logger.info("[%s] %s ok", session_id, skill)
        log_call(session_id=session_id, task_id=context.task_id, skill=skill, args=args, ok=True,
                 embedding_id=result.get("embedding_id"), duration_ms=round((time.perf_counter() - started) * 1000, 1))
        await updater.add_artifact(new_data_artifact(name=skill, data=result).parts, name=skill)
        await updater.update_status(TaskState.TASK_STATE_COMPLETED)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).update_status(
            TaskState.TASK_STATE_CANCELED
        )


def create_app() -> Starlette:
    card = build_card()
    handler = DefaultRequestHandler(
        agent_executor=TerraMindExecutor(), task_store=InMemoryTaskStore(), agent_card=card
    )
    routes = create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url="/")
    return Starlette(routes=routes)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
