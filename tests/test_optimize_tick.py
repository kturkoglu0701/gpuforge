from __future__ import annotations

from unittest.mock import patch

from gpuforge.models import GpuSnapshot, ProcessCategory, ProcessInfo, SystemSnapshot
from gpuforge.optimize import RuntimeOptimizer


def _minimal_snap() -> SystemSnapshot:
    return SystemSnapshot(
        cpu_percent=5.0,
        memory_percent=30.0,
        memory_available_mb=12000.0,
        gpu=GpuSnapshot(available=False, error="no gpu"),
        processes=[
            ProcessInfo(1, "cursor", "cursor", ProcessCategory.AI_IDE_HOST, 0.9),
            ProcessInfo(2, "rg", "rg", ProcessCategory.INDEXER, 0.9),
        ],
    )


def test_runtime_optimizer_tick_dry_run() -> None:
    cfg = {
        "ide_lean_enabled": True,
        "thresholds": {
            "gpu_util_high": 80,
            "memory_pressure_percent": 88,
            "cpu_pressure_percent": 92,
        },
        "optimizer": {"mode": "fixed"},
        "interval_seconds": 1,
    }
    opt = RuntimeOptimizer(cfg)
    with patch("gpuforge.optimize.collect_snapshot", return_value=_minimal_snap()):
        tick = opt.tick(dry_run=True, user_only=True)
    assert tick.actions_planned >= 1
    assert tick.sleep_seconds == 1.0
