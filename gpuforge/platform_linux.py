"""Native Linux is the supported runtime; other platforms are best-effort for development only."""

from __future__ import annotations

import sys


def require_linux(command: str) -> None:
    if sys.platform != "linux":
        raise SystemExit(
            f"gpuforge {command}: native Linux is required for deployment. "
            f"Current platform: {sys.platform}. "
            "Use a Linux machine or VM for production; status/once may still be used for debugging."
        )
