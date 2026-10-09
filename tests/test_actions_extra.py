from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import patch

import psutil

from gpuforge.actions import apply_action
from gpuforge.models import PolicyAction


def test_oom_and_affinity_dry_run_does_not_touch_process() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        before = open(f"/proc/{proc.pid}/oom_score_adj", encoding="utf-8").read()
        oom = apply_action(
            PolicyAction("set_oom_score_adj", proc.pid, "x", {"value": 200}),
            dry_run=True,
        )
        after = open(f"/proc/{proc.pid}/oom_score_adj", encoding="utf-8").read()
        assert oom.applied is False
        assert before == after
        aff = apply_action(
            PolicyAction("set_cpu_affinity", proc.pid, "x", {"cpus": ["0"]}),
            dry_run=True,
        )
        assert aff.applied is False
        assert aff.result["cpus"] == [0]
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_oom_write_and_affinity_apply_or_contained_error() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        path = f"/proc/{proc.pid}/oom_score_adj"
        with open(path, encoding="utf-8") as fh:
            original = int(fh.read().strip())
        target = min(1000, original + 50)
        outcome = apply_action(
            PolicyAction("set_oom_score_adj", proc.pid, "x", {"value": target}),
            dry_run=False,
        )
        assert outcome.applied is True
        with open(path, encoding="utf-8") as fh:
            assert int(fh.read().strip()) == target
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(str(original))

        cpus = psutil.Process(proc.pid).cpu_affinity()
        applied = apply_action(
            PolicyAction("set_cpu_affinity", proc.pid, "x", {"cpus": cpus}),
            dry_run=False,
        )
        assert applied.applied is True
        assert psutil.Process(proc.pid).cpu_affinity() == cpus
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_oom_and_affinity_errors_do_not_raise() -> None:
    missing = apply_action(
        PolicyAction("set_oom_score_adj", 2**22, "x", {"value": 1}),
        dry_run=False,
    )
    assert missing.applied is False
    with patch("psutil.Process.cpu_affinity", side_effect=psutil.AccessDenied(1)):
        denied = apply_action(
            PolicyAction("set_cpu_affinity", os.getpid(), "x", {"cpus": [0]}),
            dry_run=False,
        )
    assert denied.applied is False


def test_unknown_action_kind() -> None:
    out = apply_action(PolicyAction("noop", 1, "x", {}), dry_run=False)
    assert out.applied is False
    assert "unknown" in out.message
