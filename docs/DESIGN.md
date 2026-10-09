# GPUForge — product design (not runtime orchestration)

## What “orchestrator” means here

**Orchestrator** referred to the **implementation team** (architect + iterative dev passes), not to spawning worker agents on your machine at runtime. The shipped artifact is a **single self-contained package**: one daemon, one config, no bundled Ollama/Cursor/Codex binaries.

## Platform

**Native Linux first.** `gpuforge run` and production install target systemd user units on Linux. Other OSes are not supported for deployment.

## Dependencies

| Included | Not included |
|----------|----------------|
| Python package `gpuforge` + bundled `default.yaml` | Ollama, vLLM, IDE installers |
| Optional `nvidia-ml-py` for GPU metrics | Cloud LLM proxies |

GPUForge **detects** IDE and (if present) local GPU workload **by process signature only**.

## Allocation order (invariant)

1. **GPU operational** — when GPU is busy or a local GPU LLM process exists, protect compute and shed IDE background CPU (indexers, extensions).
2. **IDE session lean** — when an IDE host is open, proactively demote indexers/builds so CPU/RAM stay available for GPU work.
3. **CPU/RAM secondary** — only when memory or CPU crosses configured thresholds: demote language servers and remote LLM client processes.

## Runtime (shipped)

**Adaptive optimizer** (`gpuforge/optimize.py`) — not LLM workers:

- Dynamic sleep: fast when IDE/GPU/pressure is active; slow when idle; minimum interval on GPU utilization spikes.
- **Action cache** per PID/fingerprint so `nice`/`ionice` are not re-applied every tick.
- **Thread pool** (default 6 workers) for independent per-process applies only.

Set `optimizer.mode: fixed` and `interval_seconds` to revert to plain polling.

## Implementation phases (dev team)

| Phase | Module | Outcome |
|-------|--------|---------|
| 1 | `classify.py`, `ide.py` | IDE / workload detection |
| 2 | `allocator.py`, `policy.py`, `gpu` metrics | GPU-first rules |
| 3 | `cli.py`, `daemon.py`, `install-linux.sh` | Minimal packaging & deploy |
