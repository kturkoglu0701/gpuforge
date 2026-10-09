from __future__ import annotations

import os
import subprocess
import sys

import psutil

from gpuforge.actions import (
    ActionOutcome,
    apply_action,
    apply_actions,
    set_ionice,
    set_nice,
    suggest_env,
    suggest_env_vars,
)
from gpuforge.models import PolicyAction


def test_suggest_env_vars_merges_overrides() -> None:
    out = suggest_env_vars({"CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "8"})
    assert out["CUDA_VISIBLE_DEVICES"] == "1"
    assert out["OMP_NUM_THREADS"] == "8"


def test_dry_run_set_nice_does_not_change_process() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        before = psutil.Process(proc.pid).nice()
        outcome = set_nice(proc.pid, 10, dry_run=True)
        assert outcome.dry_run is True
        assert outcome.applied is False
        assert psutil.Process(proc.pid).nice() == before
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_apply_set_nice_same_user_can_raise_nice() -> None:
    """Unprivileged users can only increase nice (lower priority), not decrease it."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        if os.geteuid() == 0:
            outcome = set_nice(proc.pid, 5, dry_run=False)
            assert outcome.applied
        else:
            current = psutil.Process(proc.pid).nice()
            target = min(19, current + 3)
            outcome = set_nice(proc.pid, target, dry_run=False)
            assert outcome.applied
            assert psutil.Process(proc.pid).nice() == target
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_unprivileged_cannot_lower_nice_below_current() -> None:
    if os.geteuid() == 0:
        return
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        current = psutil.Process(proc.pid).nice()
        outcome = set_nice(proc.pid, current - 5, dry_run=False)
        assert outcome.applied is False
        assert "unprivileged" in outcome.message
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_suggest_env_apply_mode() -> None:
    act = PolicyAction(
        kind="suggest_env",
        pid=999,
        detail="t",
        payload={"env": {"CUDA_VISIBLE_DEVICES": "0"}},
    )
    dry = apply_action(act, dry_run=True)
    assert dry.dry_run
    assert "CUDA_VISIBLE_DEVICES" in dry.result["suggested"]
    applied = apply_action(act, dry_run=False)
    assert applied.result["suggested"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_apply_actions_batch() -> None:
    actions = [
        PolicyAction(kind="suggest_env", pid=1, detail="", payload={"env": {"OMP_NUM_THREADS": "2"}}),
    ]
    outcomes = apply_actions(actions, dry_run=True)
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], ActionOutcome)


def test_set_ionice_dry_run() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        outcome = set_ionice(proc.pid, dry_run=True)
        assert outcome.dry_run
        assert outcome.applied is False
    finally:
        proc.terminate()
        proc.wait(timeout=5)
