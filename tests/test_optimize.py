from __future__ import annotations

from gpuforge.models import GpuSnapshot, PolicyAction, ProcessCategory, ProcessInfo, SystemSnapshot
from gpuforge.optimize import (
    OptimizerRuntime,
    action_fingerprint,
    compute_sleep_interval,
    filter_redundant_actions,
)


def _snap(gpu: float = 10.0, cpu: float = 5.0, mem: float = 40.0) -> SystemSnapshot:
    return SystemSnapshot(
        cpu_percent=cpu,
        memory_percent=mem,
        memory_available_mb=8000,
        gpu=GpuSnapshot(available=True, utilization_pct=gpu),
        processes=[
            ProcessInfo(1, "cursor", "cursor", ProcessCategory.AI_IDE_HOST, 0.9),
        ],
    )


def test_fingerprint_stable() -> None:
    a = PolicyAction("set_nice", 42, "x", {"nice": 10, "rule_id": "r"})
    assert action_fingerprint(a) == action_fingerprint(a)


def test_filter_skips_repeat_actions() -> None:
    state = OptimizerRuntime()
    action = PolicyAction("set_nice", 99, "x", {"nice": 10, "rule_id": "ide"})
    state.applied_at[action_fingerprint(action)] = 1.0
    assert filter_redundant_actions([action], state) == []


def test_adaptive_fast_when_ide_active() -> None:
    cfg = {
        "optimizer": {"mode": "adaptive", "interval_active": 1.0, "interval_idle": 7.0},
        "thresholds": {
            "gpu_util_high": 80,
            "memory_pressure_percent": 88,
            "cpu_pressure_percent": 92,
        },
        "ide_lean_enabled": True,
    }
    state = OptimizerRuntime()
    sleep = compute_sleep_interval(cfg, _snap(), state)
    assert sleep == 1.0


def test_adaptive_surge_uses_min_interval() -> None:
    cfg = {
        "optimizer": {
            "mode": "adaptive",
            "interval_min": 0.5,
            "interval_idle": 6.0,
            "gpu_surge_delta": 10,
        },
        "thresholds": {
            "gpu_util_high": 80,
            "memory_pressure_percent": 88,
            "cpu_pressure_percent": 92,
        },
    }
    state = OptimizerRuntime(last_gpu_util=20.0)
    sleep = compute_sleep_interval(cfg, _snap(gpu=35.0), state)
    assert sleep == 0.5


def test_fixed_mode_uses_interval_seconds() -> None:
    cfg = {"optimizer": {"mode": "fixed"}, "interval_seconds": 4.5, "thresholds": {}}
    sleep = compute_sleep_interval(cfg, _snap(), OptimizerRuntime())
    assert sleep == 4.5
