#!/usr/bin/env bash
# Start the MCP server, the TerraMind A2A agent (if installed) and the API; Ctrl-C stops all.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p logs
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT INT TERM

python -m mcp_server.server > logs/mcp.log 2>&1 & pids+=($!)
if python -c "import terratorch" 2>/dev/null; then
  python -m terramind_agent.server > logs/terramind.log 2>&1 & pids+=($!)
else
  echo "terratorch not installed: starting without the TerraMind agent"
fi
sleep 3
uvicorn service.api:app --port 8000 > logs/api.log 2>&1 & pids+=($!)

for _ in $(seq 1 30); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  sleep 1
done
curl -sf http://127.0.0.1:8000/health && echo
echo "API http://127.0.0.1:8000  (logs in logs/). Ctrl-C to stop."
wait
