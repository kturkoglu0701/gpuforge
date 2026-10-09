from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

import psutil

from gpuforge.models import PolicyAction

# Unprivileged users may only raise nice (lower priority) on Linux without CAP_SYS_NICE.
_NICE_MIN_USER = 19
_NICE_MAX_USER = -20


@dataclass
class ActionOutcome:
    action: PolicyAction
    applied: bool
    dry_run: bool
    message: str
    result: dict[str, Any] = field(default_factory=dict)


def apply_actions(actions: list[PolicyAction], *, dry_run: bool = True) -> list[ActionOutcome]:
    return [apply_action(a, dry_run=dry_run) for a in actions]


def apply_action(action: PolicyAction, *, dry_run: bool = True) -> ActionOutcome:
    handlers = {
        "set_nice": _apply_set_nice,
        "set_ionice": _apply_set_ionice,
        "set_oom_score_adj": _apply_set_oom_score_adj,
        "set_cpu_affinity": _apply_set_cpu_affinity,
        "suggest_env": _apply_suggest_env,
    }
    handler = handlers.get(action.kind)
    if handler is None:
        return ActionOutcome(
            action=action,
            applied=False,
            dry_run=dry_run,
            message=f"unknown action kind: {action.kind}",
        )
    return handler(action, dry_run=dry_run)


def set_nice(pid: int, nice: int, *, dry_run: bool = False) -> ActionOutcome:
    act = PolicyAction(kind="set_nice", pid=pid, detail="direct", payload={"nice": nice})
    return _apply_set_nice(act, dry_run=dry_run)


def set_ionice(
    pid: int,
    ioclass: str = "idle",
    value: int = 7,
    *,
    dry_run: bool = False,
) -> ActionOutcome:
    act = PolicyAction(
        kind="set_ionice",
        pid=pid,
        detail="direct",
        payload={"class": ioclass, "value": value},
    )
    return _apply_set_ionice(act, dry_run=dry_run)


def suggest_env(
    pid: int,
    env: dict[str, str],
    *,
    dry_run: bool = False,
) -> ActionOutcome:
    act = PolicyAction(
        kind="suggest_env",
        pid=pid,
        detail="direct",
        payload={"env": env},
    )
    return _apply_suggest_env(act, dry_run=dry_run)


def _same_user(pid: int) -> bool:
    try:
        return psutil.Process(pid).uids().real == os.getuid()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _clamp_nice_for_user(nice: int, pid: int) -> tuple[int, str | None]:
    """Non-root may only increase nice (lower priority) vs current, not steal priority."""
    if os.geteuid() == 0:
        return max(_NICE_MAX_USER, min(_NICE_MIN_USER, nice)), None
    if not _same_user(pid):
        return nice, "not same user; skipping nice change"
    try:
        current = psutil.Process(pid).nice()
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        return nice, str(exc)
    if nice < current:
        return current, f"unprivileged: cannot lower nice below current ({current})"
    return max(_NICE_MAX_USER, min(_NICE_MIN_USER, nice)), None


def _apply_set_nice(action: PolicyAction, *, dry_run: bool) -> ActionOutcome:
    nice = int(action.payload["nice"])
    pid = action.pid
    clamped, note = _clamp_nice_for_user(nice, pid)
    if dry_run:
        msg = f"dry-run: would set nice={clamped} on pid {pid}"
        if note:
            msg += f" ({note})"
        return ActionOutcome(
            action=action,
            applied=False,
            dry_run=True,
            message=msg,
            result={"nice": clamped, "requested": nice},
        )
    if note and ("skipping" in note or "unprivileged" in note):
        return ActionOutcome(action=action, applied=False, dry_run=False, message=note)
    try:
        proc = psutil.Process(pid)
        proc.nice(clamped)
        return ActionOutcome(
            action=action,
            applied=True,
            dry_run=False,
            message=f"set nice={clamped} on pid {pid}",
            result={"nice": clamped},
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        return ActionOutcome(action=action, applied=False, dry_run=False, message=str(exc))


def _apply_set_ionice(action: PolicyAction, *, dry_run: bool) -> ActionOutcome:
    if sys.platform != "linux":
        return ActionOutcome(
            action=action,
            applied=False,
            dry_run=dry_run,
            message="ionice only supported on Linux",
        )
    ioclass = str(action.payload.get("class", "idle")).lower()
    value = int(action.payload.get("value", 7))
    pid = action.pid
    if dry_run:
        return ActionOutcome(
            action=action,
            applied=False,
            dry_run=True,
            message=f"dry-run: would set ionice class={ioclass} value={value} on pid {pid}",
            result={"class": ioclass, "value": value},
        )
    if hasattr(psutil.Process, "ionice"):
        try:
            proc = psutil.Process(pid)
            io_class = _psutil_ioclass(ioclass)
            if io_class == psutil.IOPRIO_CLASS_IDLE:
                proc.ionice(io_class)
            else:
                proc.ionice(io_class, value=value)
            return ActionOutcome(
                action=action,
                applied=True,
                dry_run=False,
                message=f"set ionice via psutil on pid {pid}",
                result={"class": ioclass, "value": value},
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError) as exc:
            pass
        except Exception as exc:  # noqa: BLE001 — best effort
            return ActionOutcome(action=action, applied=False, dry_run=False, message=str(exc))

    return _ionice_subprocess(pid, ioclass, value, action)


def _psutil_ioclass(name: str) -> int:
    mapping = {
        "none": psutil.IOPRIO_CLASS_NONE,
        "rt": psutil.IOPRIO_CLASS_RT,
        "be": psutil.IOPRIO_CLASS_BE,
        "idle": psutil.IOPRIO_CLASS_IDLE,
    }
    if name not in mapping:
        raise ValueError(f"unknown ionice class: {name}")
    return mapping[name]


def _ionice_subprocess(
    pid: int, ioclass: str, value: int, action: PolicyAction
) -> ActionOutcome:
    class_map = {"idle": "3", "be": "2", "rt": "1", "none": "0"}
    cls = class_map.get(ioclass, "3")
    cmd = ["ionice", "-c", cls, "-n", str(value), "-p", str(pid)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5)
        return ActionOutcome(
            action=action,
            applied=True,
            dry_run=False,
            message=f"ionice CLI ok for pid {pid}",
            result={"class": ioclass, "value": value},
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return ActionOutcome(
            action=action,
            applied=False,
            dry_run=False,
            message=f"ionice best-effort failed: {exc}",
        )


def _apply_set_oom_score_adj(action: PolicyAction, *, dry_run: bool) -> ActionOutcome:
    value = int(action.payload["value"])
    pid = action.pid
    path = f"/proc/{pid}/oom_score_adj"
    if dry_run:
        return ActionOutcome(
            action=action,
            applied=False,
            dry_run=True,
            message=f"dry-run: would write oom_score_adj={value} to {path}",
            result={"value": value},
        )
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(str(value))
        return ActionOutcome(
            action=action,
            applied=True,
            dry_run=False,
            message=f"oom_score_adj={value} for pid {pid}",
            result={"value": value},
        )
    except OSError as exc:
        return ActionOutcome(action=action, applied=False, dry_run=False, message=str(exc))


def _apply_set_cpu_affinity(action: PolicyAction, *, dry_run: bool) -> ActionOutcome:
    cpus = [int(c) for c in action.payload["cpus"]]
    pid = action.pid
    if dry_run:
        return ActionOutcome(
            action=action,
            applied=False,
            dry_run=True,
            message=f"dry-run: would set cpu_affinity={cpus} on pid {pid}",
            result={"cpus": cpus},
        )
    try:
        psutil.Process(pid).cpu_affinity(cpus)
        return ActionOutcome(
            action=action,
            applied=True,
            dry_run=False,
            message=f"cpu_affinity={cpus} on pid {pid}",
            result={"cpus": cpus},
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        return ActionOutcome(action=action, applied=False, dry_run=False, message=str(exc))


def _apply_suggest_env(action: PolicyAction, *, dry_run: bool) -> ActionOutcome:
    env = dict(action.payload.get("env", {}))
    merged = suggest_env_vars(env)
    msg = "suggest_env (launcher hints; not injected into running process)"
    if dry_run:
        msg = f"dry-run: {msg}"
    return ActionOutcome(
        action=action,
        applied=not dry_run,
        dry_run=dry_run,
        message=msg,
        result={"suggested": merged, "pid": action.pid},
    )


def suggest_env_vars(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Build CUDA / thread env suggestions for local LLM workloads."""
    base = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
    }
    if overrides:
        base.update(overrides)
    return base
