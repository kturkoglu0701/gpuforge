from __future__ import annotations

import logging
import signal
from pathlib import Path

from gpuforge.optimize import run_optimized_loop
from gpuforge.platform_linux import require_linux

log = logging.getLogger("gpuforge")
_stop = False


def _handle_signal(signum: int, _frame: object) -> None:
    global _stop
    log.info("signal %s — stopping after current tick", signum)
    _stop = True


def run_loop(
    config_path: Path | None = None,
    once: bool = False,
    dry_run: bool | None = None,
) -> None:
    global _stop
    _stop = False
    require_linux("run")
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("GPUForge adaptive optimizer starting")
    run_optimized_loop(
        config_path=config_path,
        once=once,
        dry_run=dry_run,
        stop_flag=lambda: _stop,
    )
    log.info("GPUForge loop stopped")
