from __future__ import annotations

from gpuforge.models import GpuSnapshot, PolicyAction, ProcessCategory, ProcessInfo, SystemSnapshot
from gpuforge.optimize import (
    OptimizerRuntime,
    action_fingerprint,
    collapse_actions,
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


def test_collapse_set_nice_keeps_higher_priority() -> None:
    low = PolicyAction("set_nice", 1, "a", {"nice": 15, "rule_id": "r1"})
    high = PolicyAction("set_nice", 1, "b", {"nice": -5, "rule_id": "r2"})
    out = collapse_actions([low, high])
    assert len(out) == 1
    assert out[0].payload["nice"] == -5
    # Lower nice first must not lose to a later demotion (not last-write-wins).
    out_rev = collapse_actions([high, low])
    assert out_rev[0].payload["nice"] == -5


def test_collapse_non_nice_is_last_write_wins() -> None:
    oom = collapse_actions(
        [
            PolicyAction("set_oom_score_adj", 1, "a", {"value": -900}),
            PolicyAction("set_oom_score_adj", 1, "b", {"value": 400}),
        ]
    )
    assert oom[0].payload["value"] == 400
    aff = collapse_actions(
        [
            PolicyAction("set_cpu_affinity", 1, "a", {"cpus": [0, 1]}),
            PolicyAction("set_cpu_affinity", 1, "b", {"cpus": [2, 3]}),
            PolicyAction("set_cpu_affinity", 2, "c", {"cpus": [0]}),
            PolicyAction("suggest_env", 1, "d", {"env": {"OMP_NUM_THREADS": "4"}}),
        ]
    )
    by_pid = {a.pid: a for a in aff if a.kind == "set_cpu_affinity"}
    assert by_pid[1].payload["cpus"] == [2, 3]
    assert by_pid[2].payload["cpus"] == [0]
    assert any(a.kind == "suggest_env" for a in aff)


def test_fingerprint_ignores_rule_id_for_nice() -> None:
    a = PolicyAction("set_nice", 2, "x", {"nice": 10, "rule_id": "a"})
    b = PolicyAction("set_nice", 2, "x", {"nice": 10, "rule_id": "b"})
    assert action_fingerprint(a) == action_fingerprint(b)


def test_gpu_zero_util_recorded() -> None:
    cfg = {"optimizer": {"mode": "adaptive"}, "thresholds": {}}
    state = OptimizerRuntime(last_gpu_util=None)
    snap = _snap(gpu=0.0)
    sleep = compute_sleep_interval(cfg, snap, state)
    assert sleep >= 0.5


def test_sleep_interruptible_bounds_wait(monkeypatch) -> None:
    from gpuforge.optimize import sleep_interruptible

    slept: list[float] = []
    monkeypatch.setattr("gpuforge.optimize.time.sleep", lambda s: slept.append(s))

    sleep_interruptible(5.0, lambda: True, chunk=0.25)
    assert slept == []

    flags = iter([False, True])
    sleep_interruptible(5.0, lambda: next(flags), chunk=0.25)
    assert slept == [0.25]

    slept.clear()
    sleep_interruptible(0.5, None, chunk=0.25)
    assert slept == [0.25, 0.25]


def test_fixed_mode_uses_interval_seconds() -> None:
    cfg = {"optimizer": {"mode": "fixed"}, "interval_seconds": 4.5, "thresholds": {}}
    sleep = compute_sleep_interval(cfg, _snap(), OptimizerRuntime())
    assert sleep == 4.5
