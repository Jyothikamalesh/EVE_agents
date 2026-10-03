#!/usr/bin/env bash
# One-command setup: virtualenv, dependencies (incl. the TerraMind agent), .env, model weights.
#   scripts/setup.sh                    # everything
#   scripts/setup.sh --skip-terramind   # core agent only (no torch, much smaller)
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRAS="dev,terramind"
[[ "${1:-}" == "--skip-terramind" ]] && EXTRAS="dev"

# torch/terratorch wheels exist for 3.11-3.13; prefer the newest of those
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3.13 python3.12 python3.11; do
    command -v "$c" >/dev/null 2>&1 && PY="$c" && break
  done
fi
[[ -n "$PY" ]] || { echo "Need Python 3.11-3.13 on PATH (or set PYTHON=...)"; exit 1; }
echo "Using $($PY --version) ($PY)"

[[ -d .venv ]] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
# requirements.lock pins every version used when this was tested; it constrains, it never adds
CONSTRAINTS=()
[[ -f requirements.lock ]] && CONSTRAINTS=(-c requirements.lock)
pip install "${CONSTRAINTS[@]}" -e ".[${EXTRAS}]"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env - add your GROQ_API_KEY (https://console.groq.com)."
fi

if [[ "$EXTRAS" == *terramind* ]]; then
  echo "Downloading TerraMind weights (once, ~200 MB)..."
  python - <<'PY'
from terratorch import BACKBONE_REGISTRY
BACKBONE_REGISTRY.build("terratorch_terramind_v1_tiny", pretrained=True, modalities=["S2L2A"],
                        bands={"S2L2A": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]})
print("TerraMind weights ready")
PY
fi
echo "Done. Start everything with: scripts/start_all.sh"
