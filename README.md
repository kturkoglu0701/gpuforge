# GPUForge

Self-contained **native Linux** daemon. Detects **AI IDE sessions** (Cursor, VS Code, JetBrains, …) by process patterns only — **no** bundled Ollama, Codex, or IDE installers.

## What it does

1. **IDE in use** → demote indexers and build tools so they do not clog CPU/RAM while you work.
2. **GPU busy or local GPU workload** → protect GPU compute and shed IDE background CPU.
3. **CPU/RAM pressure only when needed** → then demote language servers and remote LLM clients.

**Runtime:** an **adaptive optimizer** (default) — ~1s ticks when IDE/GPU/pressure is active, slower when idle, sub-second when GPU utilization surges; caches per-PID actions to avoid syscall thrash; applies independent tweaks in a small thread pool (not separate OS agents).

Cloud IDE models are not redirected to your GPU; this tool optimizes **your machine** when **local** GPU work matters.

## Install (minimal)

```bash
chmod +x scripts/install-linux.sh
GPUFORGE_SRC=/path/to/ai-gpu-orchestrator ./scripts/install-linux.sh
# PATH: ~/.local/bin
gpuforge status
gpuforge once --dry-run
gpuforge run
```

From a checkout:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[gpu]"
gpuforge run
```

Config override: `~/.config/gpuforge/config.yaml` (optional). Defaults ship **inside** the wheel at `gpuforge/data/default.yaml`.

Background service:

```bash
./scripts/install-user-service.sh
systemctl --user status gpuforge
```

Requires **NVIDIA driver + optional `pip install gpuforge[gpu]`** for GPU utilization metrics. Without GPU metrics, IDE-lean and CPU/RAM rules still apply.

## CLI

| Command | Purpose |
|---------|---------|
| `gpuforge run` | Daemon loop (Linux only) |
| `gpuforge status` | Snapshot |
| `gpuforge once --dry-run` | Preview policy actions |

See `docs/DESIGN.md` for allocation order and scope.

## License

MIT
