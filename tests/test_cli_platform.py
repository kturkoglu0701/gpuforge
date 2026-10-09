from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest

from gpuforge.platform_linux import require_linux


def test_gpuforge_version_cli() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    exe = root / ".venv" / "bin" / "gpuforge"
    if not exe.is_file():
        pytest.skip("venv gpuforge not installed")
    out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, check=True)
    assert "0.3" in out.stdout


def test_require_linux_exits_off_linux() -> None:
    with patch.object(sys, "platform", "darwin"):
        with pytest.raises(SystemExit):
            require_linux("run")
