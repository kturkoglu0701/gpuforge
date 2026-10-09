"""
Adaptive runtime optimizer: stateful action cache, dynamic tick rate, parallel apply.

Uses a small in-process worker pool only for independent per-PID syscalls — not separate
OS processes or LLM agents.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import psutil

from gpuforge.actions import ActionOutcome, apply_action
from gpuforge.allocator import read_signals
from gpuforge.config import load_config
from gpuforge.metrics import collect_snapshot
from gpuforge.models import PolicyAction, SystemSnapshot
from gpuforge.policy import PolicyEngine

log = logging.getLogger("gpuforge.optimize")

DEFAULT_OPTIMIZER: dict[str, Any] = {
    "mode": "adaptive",
    "interval_min": 0.5,
    "interval_max": 8.0,
    "interval_active": 1.0,
    "interval_idle": 6.0,
    "gpu_surge_delta": 12.0,
    "apply_workers": 6,
    "state_ttl_seconds": 120.0,
}

_SYSCALL_KINDS = frozenset(
    {"set_nice", "set_ionice", "set_oom_score_adj", "set_cpu_affinity"},
)


@dataclass
class OptimizerRuntime:
    last_gpu_util: float | None = None
    applied_at: dict[str, float] = field(default_factory=dict)
    last_rule_notes: tuple[str, ...] = ()


def action_fingerprint(action: PolicyAction) -> str:
    """Stable per intended syscall outcome (rule_id excluded to avoid duplicate applies)."""
    if action.kind == "suggest_env":
        rid = action.payload.get("rule_id", "")
        env = action.payload.get("env", {})
        keys = ",".join(sorted(str(k) for k in env))
        return f"suggest_env:{rid}:{keys}"
    core = (
        action.kind,
        action.pid,
        action.payload.get("nice"),
        action.payload.get("value"),
        action.payload.get("class"),
        action.payload.get("cpus"),
        action.payload.get("env"),
    )
    return "|".join(str(x) for x in core)


def collapse_actions(actions: list[PolicyAction]) -> list[PolicyAction]:
    """Merge duplicate syscalls per (kind, pid); set_nice keeps lowest nice (highest priority)."""
    passthrough: list[PolicyAction] = []
    merged: dict[tuple[str, int], PolicyAction] = {}
    for action in actions:
        if action.kind not in _SYSCALL_KINDS:
            passthrough.append(action)
            continue
        key = (action.kind, action.pid)
        existing = merged.get(key)
        if existing is None:
            merged[key] = action
            continue
        if action.kind == "set_nice":
            new_nice = int(action.payload.get("nice", 0))
            old_nice = int(existing.payload.get("nice", 0))
            if new_nice < old_nice:
                merged[key] = action
        else:
            merged[key] = action
    return passthrough + list(merged.values())


def _optimizer_cfg(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_OPTIMIZER)
    raw = config.get("optimizer")
    if isinstance(raw, dict):
        merged.update(raw)
    return merged


def prune_stale_state(state: OptimizerRuntime, ttl: float) -> None:
    """Drop cache entries for dead PIDs or expired TTL (no full process table scan)."""
    now = time.monotonic()
    for key in list(state.applied_at):
        ts = state.applied_at[key]
        if now - ts > ttl:
            state.applied_at.pop(key, None)
            continue
        if key.startswith("suggest_env:"):
            continue
        parts = key.split("|")
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if not psutil.pid_exists(pid):
            state.applied_at.pop(key, None)


def filter_redundant_actions(
    actions: list[PolicyAction],
    state: OptimizerRuntime,
) -> list[PolicyAction]:
    out: list[PolicyAction] = []
    for action in actions:
        fp = action_fingerprint(action)
        if fp in state.applied_at:
            continue
        out.append(action)
    return out


def compute_sleep_interval(
    config: dict[str, Any],
    snapshot: SystemSnapshot,
    state: OptimizerRuntime,
) -> float:
    opt = _optimizer_cfg(config)
    if opt.get("mode") == "fixed":
        return float(config.get("interval_seconds", 5))

    t = config.get("thresholds", {})
    signals = read_signals(
        snapshot,
        gpu_util_high=float(t.get("gpu_util_high", 55)),
        memory_pressure_percent=float(t.get("memory_pressure_percent", 88)),
        cpu_pressure_percent=float(t.get("cpu_pressure_percent", 92)),
    )
    gpu_now = snapshot.gpu.utilization_pct if snapshot.gpu.available else None

    if signals.ide_active or signals.llm_gpu_active or signals.gpu_busy:
        interval = float(opt["interval_active"])
    elif signals.ram_pressure or signals.cpu_pressure:
        interval = float(opt["interval_active"])
    else:
        interval = float(opt["interval_idle"])

    if state.last_gpu_util is not None and gpu_now is not None:
        if abs(gpu_now - state.last_gpu_util) >= float(opt["gpu_surge_delta"]):
            interval = float(opt["interval_min"])

    lo = float(opt["interval_min"])
    hi = float(opt["interval_max"])
    return max(lo, min(hi, interval))


def apply_actions_concurrent(
    actions: list[PolicyAction],
    *,
    dry_run: bool,
    max_workers: int,
) -> list[ActionOutcome]:
    if not actions:
        return []
    if dry_run or len(actions) == 1:
        return [apply_action(a, dry_run=dry_run) for a in actions]

    workers = max(1, min(max_workers, len(actions)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(apply_action, a, dry_run=False) for a in actions]
        return [fut.result() for fut in futures]


@dataclass
class TickResult:
    snapshot: SystemSnapshot
    notes: tuple[str, ...]
    actions_planned: int
    actions_applied: int
    sleep_seconds: float


class RuntimeOptimizer:
    """One daemon tick: measure → decide → apply (cached) → schedule next wake."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.policy = PolicyEngine(config)
        self.state = OptimizerRuntime()
        self._fast_cpu = False
        self._lock = threading.Lock()

    def tick(self, *, dry_run: bool, user_only: bool) -> TickResult:
        opt = _optimizer_cfg(self.config)
        with self._lock:
            prune_stale_state(self.state, float(opt["state_ttl_seconds"]))

        snap = collect_snapshot(user_only=user_only, fast_cpu=self._fast_cpu)
        self._fast_cpu = True

        result = self.policy.evaluate(snap)
        notes = tuple(result.notes)
        with self._lock:
            if notes != self.state.last_rule_notes:
                for note in notes:
                    if note not in self.state.last_rule_notes:
                        log.info("rule active: %s", note)
                self.state.last_rule_notes = notes

        planned = collapse_actions(result.actions)
        with self._lock:
            to_apply = filter_redundant_actions(planned, self.state)
        outcomes = apply_actions_concurrent(
            to_apply,
            dry_run=dry_run,
            max_workers=int(opt["apply_workers"]),
        )
        now = time.monotonic()
        with self._lock:
            for action, outcome in zip(to_apply, outcomes, strict=True):
                if outcome.applied and not dry_run:
                    self.state.applied_at[action_fingerprint(action)] = now
                if outcome.applied or dry_run or "dry-run" in outcome.message:
                    log.info("%s", outcome.message)

            if snap.gpu.available and snap.gpu.utilization_pct is not None:
                self.state.last_gpu_util = snap.gpu.utilization_pct

            sleep_s = compute_sleep_interval(self.config, snap, self.state)

        return TickResult(
            snapshot=snap,
            notes=notes,
            actions_planned=len(planned),
            actions_applied=sum(1 for o in outcomes if o.applied),
            sleep_seconds=sleep_s,
        )


def run_optimized_loop(
    config_path: Any = None,
    once: bool = False,
    dry_run: bool | None = None,
    stop_flag: Callable[[], bool] | None = None,
) -> None:
    from pathlib import Path

    cfg = load_config(Path(config_path) if config_path else None)
    if dry_run is not None:
        cfg["dry_run"] = dry_run
    is_dry = bool(cfg.get("dry_run", False))
    user_only = bool(cfg.get("user_only", True))
    opt = _optimizer_cfg(cfg)

    runtime = RuntimeOptimizer(cfg)
    log.info(
        "optimizer mode=%s dry_run=%s user_only=%s",
        opt.get("mode"),
        is_dry,
        user_only,
    )

    while True:
        if stop_flag and stop_flag():
            break
        tick = runtime.tick(dry_run=is_dry, user_only=user_only)
        log.debug(
            "tick planned=%s applied=%s sleep=%.2fs gpu=%s",
            tick.actions_planned,
            tick.actions_applied,
            tick.sleep_seconds,
            tick.snapshot.gpu.utilization_pct,
        )
        if once:
            break
        time.sleep(tick.sleep_seconds)
