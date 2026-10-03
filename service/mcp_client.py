"""Connects to the (single) EO tools MCP server and returns traced, agent-ready tools.

Kept separate from ``service/api.py`` so the connection/tracing wiring can
be unit-tested or reused (e.g. from the demo script) without booting FastAPI.

Everything the agent can do with satellite data and geography comes through this one MCP
server (``mcp_server/server.py``). The TerraMind foundation-model capability is not an MCP
server: it is a separate *agent* reached over A2A (``service/a2a_client.py``).
"""

import logging
import os
from typing import List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from .tracing import wrap_tool_with_tracing

logger = logging.getLogger(__name__)

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8765/mcp")


async def load_traced_tools(session_id_getter, extra_tools: List[BaseTool] | None = None) -> List[BaseTool]:
    """Connect to the EO tools MCP server and return tools wrapped for tracing.

    *session_id_getter* is a zero-arg callable the wrapped tools call at
    invocation time to find out which session's request they're running
    inside (set via a contextvar in ``service.api`` before each graph call).
    """
    client = MultiServerMCPClient({"eo_tools": {"url": MCP_SERVER_URL, "transport": "streamable_http"}})
    tools = await client.get_tools(server_name="eo_tools")  # required: let this raise if the server is down

    return [wrap_tool_with_tracing(t, session_id_getter) for t in tools + list(extra_tools or [])]
