from unittest.mock import patch

import gpuforge.metrics as metrics
from gpuforge.metrics import read_gpu


def _fake_nvml(*, util: int = 42, init_raises: Exception | None = None, init_calls: list | None = None):
    class _Util:
        gpu = util

    class _Mem:
        used = 1024 * 1024 * 100
        total = 1024 * 1024 * 8000

    fake = type("NV", (), {})()

    def nvml_init() -> None:
        if init_calls is not None:
            init_calls.append(1)
        if init_raises is not None:
            raise init_raises

    fake.nvmlInit = nvml_init
    fake.nvmlShutdown = lambda: None
    fake.nvmlDeviceGetHandleByIndex = lambda _i: object()
    fake.nvmlDeviceGetName = lambda _h: b"TestGPU"
    fake.nvmlDeviceGetUtilizationRates = lambda _h: _Util()
    fake.nvmlDeviceGetMemoryInfo = lambda _h: _Mem()
    return fake


def test_read_gpu_success_records_megabytes_and_inits_once() -> None:
    calls: list[int] = []
    fake = _fake_nvml(init_calls=calls)
    metrics._nvml_ready = False
    with patch.dict("sys.modules", {"pynvml": fake}):
        first = read_gpu()
        second = read_gpu()
    assert first.available and second.available
    assert first.memory_used_mb == 100
    assert first.memory_total_mb == 8000
    assert calls == [1]
    metrics._nvml_ready = False


def test_read_gpu_zero_util_stays_available() -> None:
    fake = _fake_nvml(util=0)
    metrics._nvml_ready = False
    with patch.dict("sys.modules", {"pynvml": fake}):
        snap = read_gpu()
    assert snap.available is True
    assert snap.utilization_pct == 0.0
    metrics._nvml_ready = False


def test_read_gpu_failed_init_does_not_stick() -> None:
    metrics._nvml_ready = False
    with patch.dict("sys.modules", {"pynvml": _fake_nvml(init_raises=RuntimeError("no driver"))}):
        snap = read_gpu()
    assert snap.available is False
    assert metrics._nvml_ready is False


def test_read_gpu_already_initialized_still_reads() -> None:
    metrics._nvml_ready = False
    fake = _fake_nvml(init_raises=RuntimeError("NVML_ERROR_ALREADY_INITIALIZED"))
    with patch.dict("sys.modules", {"pynvml": fake}):
        snap = read_gpu()
    assert snap.available is True
    assert snap.utilization_pct == 42.0
    metrics._nvml_ready = False


def test_read_gpu_graceful_without_driver() -> None:
    metrics._nvml_ready = False
    with patch.dict("sys.modules", {"pynvml": None}):
        snap = read_gpu()
    assert snap.available is False
    assert snap.error is not None
    assert metrics._nvml_ready is False
