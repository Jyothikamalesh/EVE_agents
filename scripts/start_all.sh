#!/usr/bin/env bash
# Start everything: MCP server, TerraMind A2A agent, API (hosts the EO LangGraph agent) and the UI.
# Ctrl-C stops all.
#   scripts/start_all.sh                    # all four
#   scripts/start_all.sh --skip-terramind   # no TerraMind agent (core agent only)
#   scripts/start_all.sh --no-ui            # no Streamlit UI
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate

SKIP_TERRAMIND=0
NO_UI=0
for arg in "$@"; do
  case "$arg" in
    --skip-terramind) SKIP_TERRAMIND=1 ;;
    --no-ui) NO_UI=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

if [[ "$SKIP_TERRAMIND" == 0 ]] && ! python -c "import terratorch" 2>/dev/null; then
  echo "terratorch is not installed, so the TerraMind agent cannot start."
  echo "Run scripts/setup.sh to install it, or pass --skip-terramind to run without it."
  exit 1
fi

mkdir -p logs
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT INT TERM

wait_for() { # name url
  for _ in $(seq 1 60); do
    curl -sf "$2" >/dev/null 2>&1 && { echo "  up: $1"; return 0; }
    sleep 1
  done
  echo "  FAILED to start: $1 (see logs/)"
  return 1
}

echo "Starting..."
python -m mcp_server.server > logs/mcp.log 2>&1 & pids+=($!)

if [[ "$SKIP_TERRAMIND" == 0 ]]; then
  python -m terramind_agent.server > logs/terramind.log 2>&1 & pids+=($!)
  # The EO agent discovers the TerraMind agent's card when the API starts, so wait for it first.
  wait_for "TerraMind A2A agent (8767)" "http://127.0.0.1:8767/.well-known/agent-card.json"
fi
sleep 2  # MCP server

uvicorn service.api:app --port 8000 > logs/api.log 2>&1 & pids+=($!)
wait_for "API + EO agent (8000)" "http://127.0.0.1:8000/health"

if [[ "$NO_UI" == 0 ]]; then
  streamlit run scripts/ui_app.py --server.port 8501 --server.headless true \
    > logs/ui.log 2>&1 & pids+=($!)
  wait_for "UI (8501)" "http://127.0.0.1:8501/_stcore/health"
fi

echo
echo "MCP server        http://127.0.0.1:8765/mcp"
[[ "$SKIP_TERRAMIND" == 0 ]] && echo "TerraMind agent   http://127.0.0.1:8767"
echo "API + EO agent    http://127.0.0.1:8000"
[[ "$NO_UI" == 0 ]] && echo "UI                http://127.0.0.1:8501"
echo "Logs in logs/. Ctrl-C to stop."
wait
