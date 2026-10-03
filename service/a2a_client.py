"""Calls the TerraMind A2A agent and exposes its skills as LangChain tools.

The EO agent stays a plain tool-calling graph: ``embed_scene``, ``compare_embeddings`` and
``rank_similar`` are ordinary tools to it, each implemented as one A2A message to the remote
TerraMind agent (``terramind_agent/server.py``), whose skills are discovered from its Agent
Card. Optional by design: if the remote agent isn't running at startup, the service logs a
warning and continues with the MCP tools only.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers.proto_helpers import get_data_parts, get_message_text, new_data_message
from a2a.types.a2a_pb2 import Role, SendMessageRequest, TaskState
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

TERRAMIND_A2A_URL = os.environ.get("TERRAMIND_A2A_URL", "http://127.0.0.1:8767")
CALL_TIMEOUT_S = float(os.environ.get("TERRAMIND_A2A_TIMEOUT", "180"))


def _intify(obj: Any) -> Any:
    """A2A data parts are protobuf Structs, which turn every number into a float (224 -> 224.0)."""
    if isinstance(obj, dict):
        return {k: _intify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_intify(v) for v in obj]
    return int(obj) if isinstance(obj, float) and obj.is_integer() else obj


class A2AError(RuntimeError):
    """The remote agent failed the task or returned nothing usable."""


async def call_skill(
    base_url: str, skill: str, args: Dict[str, Any], timeout: float = CALL_TIMEOUT_S,
    http_client: httpx.AsyncClient | None = None, session_id: str | None = None,
) -> Dict[str, Any]:
    """Send one skill call as an A2A message and return the artifact's JSON.

    *http_client* is for tests (e.g. an in-process ASGI transport)."""
    async with (http_client or httpx.AsyncClient(timeout=timeout)) as http:
        client = await ClientFactory(ClientConfig(streaming=False, httpx_client=http)).create_from_url(base_url)
        message = new_data_message({"skill": skill, "args": args}, role=Role.ROLE_USER)
        message.message_id = str(uuid.uuid4())
        if session_id:  # lets the remote agent's logs and stored embeddings be traced to the chat session
            message.context_id = session_id
            message.metadata.update({"session_id": session_id})
        task = None
        async for resp in client.send_message(SendMessageRequest(message=message)):
            if resp.HasField("task"):
                task = resp.task
            elif resp.HasField("message"):
                raise A2AError(get_message_text(resp.message) or "agent replied with a message and no result")
    if task is None:
        raise A2AError("no task returned")
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        detail = get_message_text(task.status.message) if task.status.HasField("message") else ""
        raise A2AError(f"task {TaskState.Name(task.status.state)}: {detail}".strip())
    for artifact in task.artifacts:
        data = get_data_parts(artifact.parts)
        if data:
            return _intify(data[0])
    raise A2AError("task completed without a data artifact")


class _EmbedArgs(BaseModel):
    collection: str = Field(description="STAC collection id, e.g. 'sentinel-2-l2a' (from search_stac_items).")
    item_id: str = Field(description="STAC item id of the scene to embed (from search_stac_items).")
    patch_size: int = Field(default=224, description="Side of the centred crop in pixels, multiple of 16 (32-512).")


class _CompareArgs(BaseModel):
    id_a: str = Field(description="embedding_id of the first scene (from embed_scene).")
    id_b: str = Field(description="embedding_id of the second scene (from embed_scene).")


class _RankArgs(BaseModel):
    reference_id: str = Field(description="embedding_id of the reference scene.")
    candidate_ids: List[str] = Field(description="embedding_ids of the scenes to rank against the reference.")


_TOOLS = {
    "embed_scene": (
        _EmbedArgs,
        "Run a Sentinel-2 scene through the TerraMind foundation model (remote A2A agent). Returns an "
        "embedding_id plus the scene's date, cloud cover, tile id and basic stats; the full tensor stays "
        "server-side. Use only when the user asks to embed, analyse or compare imagery at a representation "
        "level, after search_stac_items has given a collection and item_id.",
    ),
    "compare_embeddings": (
        _CompareArgs,
        "Compare two embedded scenes: overall cosine similarity, and where they differ (per-tile "
        "mean/min/max and the 3 least-similar grid cells) when both share a tile id. Needs embedding_ids "
        "from embed_scene. Useful for change detection between dates over the same tile.",
    ),
    "rank_similar": (
        _RankArgs,
        "Rank several embedded scenes by similarity to a reference scene. Needs embedding_ids from embed_scene.",
    ),
}


def _make_tool(name: str, schema: type[BaseModel], description: str, base_url: str, session_id_getter) -> StructuredTool:
    async def _run(**kwargs: Any) -> str:
        try:
            return json.dumps(await call_skill(base_url, name, kwargs, session_id=session_id_getter()))
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error by the tools node
            raise RuntimeError(f"TerraMind agent: {exc}") from exc

    return StructuredTool.from_function(coroutine=_run, name=name, description=description, args_schema=schema)


async def load_a2a_tools(base_url: str = TERRAMIND_A2A_URL, session_id_getter=lambda: None) -> List[BaseTool]:
    """Tools for every skill the remote agent advertises; [] (with a warning) if it's unreachable."""
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            card = await A2ACardResolver(http, base_url).get_agent_card()
            skills = {s.id for s in card.skills}
    except Exception as exc:  # noqa: BLE001
        logger.warning("TerraMind A2A agent not reachable at %s (%s); continuing without it.", base_url, exc)
        return []
    tools = [_make_tool(n, s, d, base_url, session_id_getter) for n, (s, d) in _TOOLS.items() if n in skills]
    logger.info("TerraMind A2A agent at %s: tools %s", base_url, [t.name for t in tools])
    return tools
