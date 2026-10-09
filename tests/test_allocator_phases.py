from gpuforge.allocator import AllocationPhase, active_phases
from gpuforge.allocator import AllocationSignals


def test_active_phases_gpu_and_ide() -> None:
    sig = AllocationSignals(
        ide_active=True,
        llm_gpu_active=True,
        gpu_busy=True,
        ram_pressure=False,
        cpu_pressure=False,
    )
    phases = active_phases(sig, ide_lean_enabled=True)
    assert AllocationPhase.GPU_OPERATIONAL in phases
    assert AllocationPhase.IDE_LEAN_BACKGROUND in phases
