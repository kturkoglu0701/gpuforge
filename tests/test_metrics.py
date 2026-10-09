from unittest.mock import patch

from gpuforge.metrics import read_gpu


def test_read_gpu_graceful_without_driver() -> None:
    with patch.dict("sys.modules", {"pynvml": None}):
        snap = read_gpu()
    assert snap.available is False
    assert snap.error is not None
