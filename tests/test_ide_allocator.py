from gpuforge.allocator import read_signals
from gpuforge.ide import ide_session_active
from gpuforge.models import GpuSnapshot, ProcessCategory, ProcessInfo, SystemSnapshot


def test_ide_session_active() -> None:
    snap = SystemSnapshot(
        cpu_percent=1,
        memory_percent=1,
        memory_available_mb=1000,
        gpu=GpuSnapshot(available=False),
        processes=[ProcessInfo(1, "cursor", "cursor", ProcessCategory.AI_IDE_HOST, 0.9)],
    )
    assert ide_session_active(snap) is True


def test_signals_ide_without_pressure() -> None:
    snap = SystemSnapshot(
        cpu_percent=10,
        memory_percent=40,
        memory_available_mb=4000,
        gpu=GpuSnapshot(available=True, utilization_pct=10),
        processes=[ProcessInfo(1, "code", "code", ProcessCategory.AI_IDE_HOST, 0.9)],
    )
    sig = read_signals(snap, gpu_util_high=80, memory_pressure_percent=88, cpu_pressure_percent=92)
    assert sig.ide_active
    assert not sig.ram_pressure
    assert not sig.cpu_pressure
