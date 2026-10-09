"""Shared action list normalization (policy + runtime optimizer)."""

from __future__ import annotations

from gpuforge.models import PolicyAction

_SYSCALL_KINDS = frozenset(
    {"set_nice", "set_ionice", "set_oom_score_adj", "set_cpu_affinity"},
)


def collapse_actions(actions: list[PolicyAction]) -> list[PolicyAction]:
    """Merge duplicate syscalls per (kind, pid); set_nice keeps lowest nice (highest priority)."""
    passthrough: list[PolicyAction] = []
    merged: dict[tuple[str, int], PolicyAction] = {}
    for action in actions:
        if action.kind not in _SYSCALL_KINDS:
            passthrough.append(action)
            continue
        key = (action.kind, action.pid)
        existing = merged.get(key)
        if existing is None:
            merged[key] = action
            continue
        if action.kind == "set_nice":
            new_nice = int(action.payload.get("nice", 0))
            old_nice = int(existing.payload.get("nice", 0))
            if new_nice < old_nice:
                merged[key] = action
        else:
            merged[key] = action
    return passthrough + list(merged.values())
