"""Backend wiring for the Tier 1 EO agent: MCP client, tracing, FastAPI.

Kept outside ``agents/`` deliberately — that package is the portable,
backend-agnostic graph library; this is the one app that consumes it,
connecting it to the MCP server over HTTP and exposing it over HTTP in turn.
"""
