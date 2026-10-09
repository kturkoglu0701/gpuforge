# GPUForge — code audit (approval gate)

**Reviewer stance:** systems software / OS scheduling (20+ yr).  
**Scope:** `main` @ v0.3.x — user-space Linux priority orchestrator (not numerical physics).

## Executive verdict

| Area | Status | Notes |
|------|--------|-------|
| **Segfault risk** | Low | Pure Python + optional NVML; no native extension code in-tree |
| **Race conditions** | Mitigated | NVML under lock; optimizer state under `threading.Lock`; per-PID collapse before parallel apply |
| **Algorithmic correctness** | Acceptable with limits | Heuristic classification + nice/ionice; cannot boost priority without `CAP_SYS_NICE` |
| **Mathematical derivations** | N/A | No orbital/RF theory; thresholds are configurable engineering constants |
| **Test coverage** | **76%** line on core modules (`cli`/`daemon` entrypoints omitted) | 46 tests; gate `cov-fail-under=68` |
| **Approval** | **Conditional pass** | Suitable for POC / personal Linux use; not production hardening for untrusted multi-tenant |

## Findings addressed in this audit

1. **GPU util 0% treated as false** — fixed (`is not None` check).
2. **Duplicate `set_nice` per PID across rules** — `collapse_actions()` keeps lowest nice.
3. **Fingerprint included `rule_id`** — caused redundant syscalls; removed for syscall kinds.
4. **NVML init/shutdown per tick** — replaced with locked one-time init + `atexit` shutdown (thread-safe).
5. **Full `process_iter` every prune** — replaced with `pid_exists` on cached PIDs only.
6. **Dead code** — removed `gpu.py` shim and unused `_gpu_util_high`.
7. **Unused import** — `as_completed` removed from optimizer.

## Residual risks (documented, not bugs)

- **Cloud LLM traffic** is not schedulable locally.
- **Unprivileged nice lowering** for GPU protect rules may no-op; logged via `actions.py`.
- **Parallel `apply_action`** on different PIDs is safe; same-PID collapsed to one syscall per kind.
- **Signal handler + sleep loop** — cooperative shutdown only; no re-entrancy in NVML from workers.

## Coverage gap analysis

| Module | Risk | Gap / mitigation |
|--------|------|------------------|
| `cli.py` | UX | Smoke via `--version`; full CLI integration optional |
| `daemon.py` | Signals | Manual/systemd test; thin wrapper over optimizer |
| `actions.py` | Privilege edge cases | Partial; ionice/oom/affinity need root/CAP tests on CI runner |
| `classify.py` | Misclassification | Pattern tests; not ML-grade ground truth |
| `metrics.py` | NVML | Mock import failure test added |
| `optimize.py` | Concurrency | Collapse/fingerprint/prune unit tests |

## Keywords (review fan-out)

`concurrency`, `scheduling`, `coverage`, `heuristics`, `NVML`

## References

Linux scheduler: nice −20…19, ionice classes (cf. `man 1 nice`, `man 1 ionice`).  
GPU metrics: NVIDIA NVML utilization API (engineering readout, not a theorem).
