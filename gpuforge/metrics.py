from __future__ import annotations

import psutil

from gpuforge.classify import scan_processes
from gpuforge.models import GpuSnapshot, SystemSnapshot


def read_gpu() -> GpuSnapshot:
    """NVML metrics when nvidia-ml-py is installed; graceful fallback otherwise."""
    try:
        import pynvml  # type: ignore[import-untyped]  # nvidia-ml-py

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return GpuSnapshot(
                available=True,
                name=name,
                utilization_pct=float(util.gpu),
                memory_used_mb=mem.used / (1024 * 1024),
                memory_total_mb=mem.total / (1024 * 1024),
            )
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001 — optional GPU stack
        return GpuSnapshot(available=False, error=str(exc))


def collect_snapshot(user_only: bool = True, *, fast_cpu: bool = False) -> SystemSnapshot:
    vm = psutil.virtual_memory()
    cpu_interval = None if fast_cpu else 0.15
    if cpu_interval is None:
        cpu_pct = psutil.cpu_percent(interval=None)
    else:
        cpu_pct = psutil.cpu_percent(interval=cpu_interval)
    return SystemSnapshot(
        cpu_percent=cpu_pct,
        memory_percent=vm.percent,
        memory_available_mb=vm.available / (1024 * 1024),
        gpu=read_gpu(),
        processes=scan_processes(user_only=user_only),
    )
