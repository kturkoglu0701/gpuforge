from unittest.mock import patch

from gpuforge.metrics import read_gpu


def test_read_gpu_success_mocked() -> None:
    class _Util:
        gpu = 42

    class _Mem:
        used = 1024 * 1024 * 100
        total = 1024 * 1024 * 8000

    fake = type("NV", (), {})()
    fake.nvmlInit = lambda: None
    fake.nvmlShutdown = lambda: None
    fake.nvmlDeviceGetHandleByIndex = lambda _i: object()
    fake.nvmlDeviceGetName = lambda _h: b"TestGPU"
    fake.nvmlDeviceGetUtilizationRates = lambda _h: _Util()
    fake.nvmlDeviceGetMemoryInfo = lambda _h: _Mem()

    import gpuforge.metrics as metrics

    metrics._nvml_ready = False
    with patch.dict("sys.modules", {"pynvml": fake}):
        snap = read_gpu()
    assert snap.available is True
    assert snap.name == "TestGPU"
    assert snap.utilization_pct == 42.0
    metrics._nvml_ready = False
    with patch.dict("sys.modules", {"pynvml": None}):
        snap = read_gpu()
    assert snap.available is False
    assert snap.error is not None
