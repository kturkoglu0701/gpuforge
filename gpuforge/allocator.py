"""
Resource allocation order (product invariant):

1. GPU compute — keep local GPU inference/render fed; shed CPU hogs when GPU is busy.
2. IDE interactive — host process stays at baseline priority.
3. CPU / RAM — adjust extension and remote-client priority only under pressure.

This module encodes that order for the policy engine; it does not run sub-agents or external tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from gpuforge.ide import ide_session_active
from gpuforge.models import ProcessCategory, SystemSnapshot


class AllocationPhase(IntEnum):
    GPU_OPERATIONAL = 1
    IDE_LEAN_BACKGROUND = 2
    CPU_RAM_SECONDARY = 3


@dataclass(frozen=True)
class AllocationSignals:
    ide_active: bool
    llm_gpu_active: bool
    gpu_busy: bool
    ram_pressure: bool
    cpu_pressure: bool


def read_signals(
    snapshot: SystemSnapshot,
    *,
    gpu_util_high: float,
    memory_pressure_percent: float,
    cpu_pressure_percent: float,
) -> AllocationSignals:
    gpu = snapshot.gpu
    gpu_busy = (
        gpu.available
        and gpu.utilization_pct is not None
        and gpu.utilization_pct >= gpu_util_high
    )
    return AllocationSignals(
        ide_active=ide_session_active(snapshot),
        llm_gpu_active=any(p.category == ProcessCategory.LLM_LOCAL_GPU for p in snapshot.processes),
        gpu_busy=gpu_busy,
        ram_pressure=snapshot.memory_percent >= memory_pressure_percent,
        cpu_pressure=snapshot.cpu_percent >= cpu_pressure_percent,
    )


def active_phases(signals: AllocationSignals, *, ide_lean_enabled: bool) -> list[AllocationPhase]:
    phases: list[AllocationPhase] = []
    if signals.llm_gpu_active or signals.gpu_busy:
        phases.append(AllocationPhase.GPU_OPERATIONAL)
    if ide_lean_enabled and signals.ide_active:
        phases.append(AllocationPhase.IDE_LEAN_BACKGROUND)
    if signals.ram_pressure or signals.cpu_pressure:
        phases.append(AllocationPhase.CPU_RAM_SECONDARY)
    return phases
