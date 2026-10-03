"""EOReactAgent — ``ReactAgent`` with a system prompt scoped to the Tier 1 EO tools.

``agents.graphs.react.ReactAgent`` ships the full EVE/Phi-lab identity prompt,
which assumes an ``eve_retrieval_retrieve`` RAG tool this service doesn't
register. Subclassing only to point ``AgentGraph._prompts_yaml_path`` (which
resolves relative to the *subclass's* module file) at a prompt scoped to the
tools this service actually has — geocoding, STAC search, weather — with
explicit groundedness instructions. No other behaviour changes: the
tool-calling loop, retry/fallback policy, and checkpointing all come
unmodified from ``ReactAgent``.
"""

from agents.graphs.react.graph import ReactAgent


class EOReactAgent(ReactAgent):
    name = "eo_react"
