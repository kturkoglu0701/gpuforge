"""IDE session detection — no third-party bundles, pattern-only."""

from __future__ import annotations

from gpuforge.models import ProcessCategory, ProcessInfo, SystemSnapshot


def ide_session_active(snapshot: SystemSnapshot) -> bool:
    """True when a primary AI IDE host process is running (Cursor, VS Code, etc.)."""
    return any(p.category == ProcessCategory.AI_IDE_HOST for p in snapshot.processes)


def ide_host_processes(snapshot: SystemSnapshot) -> list[ProcessInfo]:
    return [p for p in snapshot.processes if p.category == ProcessCategory.AI_IDE_HOST]
