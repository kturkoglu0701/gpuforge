#!/usr/bin/env bash
# Install GPUForge as a systemd *user* service (Linux). WSL2: enable linger if needed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

VENV="${REPO_ROOT}/.venv"
if [[ ! -x "${VENV}/bin/gpuforge" ]]; then
  python3 -m venv "$VENV"
  "${VENV}/bin/pip" install -e "${REPO_ROOT}[gpu,dev]"
fi

cat > "${UNIT_DIR}/gpuforge.service" <<EOF
[Unit]
Description=GPUForge AI IDE / GPU orchestrator
After=default.target

[Service]
Type=simple
ExecStart=${VENV}/bin/gpuforge run
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now gpuforge.service
echo "GPUForge user service installed. Check: systemctl --user status gpuforge"
echo ""
echo "WSL2: loginctl enable-linger \"\$USER\"  # keep user service running without a login session"
echo "macOS: use launchd LaunchAgent instead of systemd — see README.md"
