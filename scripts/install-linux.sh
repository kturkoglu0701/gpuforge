#!/usr/bin/env bash
# Minimal native-Linux install — one venv, one package, no Ollama/Cursor bundles.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "GPUForge installs on native Linux only." >&2
  exit 1
fi

INSTALL_ROOT="${GPUFORGE_HOME:-$HOME/.local/share/gpuforge}"
VENV="${INSTALL_ROOT}/venv"
mkdir -p "$INSTALL_ROOT"

if [[ -n "${GPUFORGE_SRC:-}" ]]; then
  SRC="$GPUFORGE_SRC"
else
  SRC="$(cd "$(dirname "$0")/.." && pwd)"
fi

python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -e "${SRC}[gpu]"

mkdir -p "$HOME/.local/bin"
ln -sf "${VENV}/bin/gpuforge" "$HOME/.local/bin/gpuforge"

echo "Installed: gpuforge -> ${VENV}/bin/gpuforge"
echo "Ensure ~/.local/bin is on PATH, then: gpuforge run"
echo "Optional user service: ${SRC}/scripts/install-user-service.sh"
