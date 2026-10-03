"""EXPERIMENT ONLY, not connected to the service: nothing in ``service/api.py`` imports this.

Kept to show the Qdrant-local approach that was evaluated for cross-session memory and set
aside for now (the conversation is carried by the checkpointer + rolling summary, see README).
To try it, call ``recall``/``remember`` from the API and add ``qdrant-client`` and ``fastembed``
(not in the default install).

Per-session semantic memory: durable facts extracted from a conversation, retrieved by similarity.

Separate from the checkpointer on purpose. The checkpointer holds the full transcript
(and gets trimmed for the model). Facts are the small set of things worth remembering
that should survive trimming: "the user's favorite city is Hyderabad", "the user is
comparing Sentinel-2 tiles 43QHV and 44QKE". Each turn, the facts most similar to the
new message are retrieved and injected into the system prompt.

Storage is Qdrant in local mode (a directory on disk, no server). Local mode takes a
file lock, so run a single API worker against it.
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import List

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

logger = logging.getLogger(__name__)

COLLECTION = "session_facts"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_NAMESPACE = uuid.UUID("6c1e7a52-0000-4000-8000-000000000000")

_EXTRACT_PROMPT = """You maintain a short memory of durable facts about one user's session.

Known facts so far:
{known}

Latest exchange:
User: {user}
Assistant: {reply}

List NEW durable facts from this exchange that would still matter in a later turn: the user's
preferences, places they care about, dates or ranges they chose, identifiers they are working
with, and conclusions the assistant reached. Skip small talk, one-off tool outputs, and anything
already in the known facts. Each fact must be one short standalone sentence.

Respond with only a JSON array of strings, e.g. ["..."]. Respond with [] if nothing is new."""


class SessionMemory:
    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(path))
        self._embedder = TextEmbedding(EMBED_MODEL)
        if not self._client.collection_exists(COLLECTION):
            self._client.create_collection(
                COLLECTION, vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._embedder.embed(texts)]

    def facts(self, session_id: str) -> List[str]:
        points, _ = self._client.scroll(
            COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]),
            limit=1000,
            with_payload=True,
        )
        return [p.payload["fact"] for p in points]

    def recall(self, session_id: str, query: str, k: int = 5) -> List[str]:
        vector = self._embed([query])[0]
        hits = self._client.query_points(
            COLLECTION,
            query=vector,
            query_filter=Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]),
            limit=k,
        ).points
        return [h.payload["fact"] for h in hits]

    def remember(self, session_id: str, new_facts: List[str]) -> int:
        existing = {f.lower() for f in self.facts(session_id)}
        fresh = [f.strip() for f in new_facts if f.strip() and f.strip().lower() not in existing]
        if not fresh:
            return 0
        vectors = self._embed(fresh)
        self._client.upsert(
            COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid5(_NAMESPACE, f"{session_id}:{fact.lower()}")),
                    vector=vec,
                    payload={"session_id": session_id, "fact": fact},
                )
                for fact, vec in zip(fresh, vectors)
            ],
        )
        return len(fresh)


def parse_fact_list(raw: str) -> List[str]:
    """Pull a JSON array of strings out of a model reply, tolerating surrounding prose."""
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [s for s in data if isinstance(s, str)] if isinstance(data, list) else []


async def extract_facts(llm, memory: SessionMemory, session_id: str, user: str, reply: str) -> List[str]:
    known = memory.facts(session_id)
    prompt = _EXTRACT_PROMPT.format(
        known="\n".join(f"- {f}" for f in known) or "(none yet)",
        user=user,
        reply=reply,
    )
    response = await llm.ainvoke([("user", prompt)])
    return parse_fact_list(str(response.content))


def format_for_prompt(facts: List[str]) -> str | None:
    if not facts:
        return None
    lines = "\n".join(f"- {f}" for f in facts)
    return (
        "Durable facts about this user's session (use them when relevant; they came from earlier turns):\n"
        f"{lines}"
    )
