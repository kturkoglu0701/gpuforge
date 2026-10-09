from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ProcessCategory(StrEnum):
    AI_IDE_HOST = "ai_ide_host"
    AI_IDE_EXTENSION = "ai_ide_extension"
    LLM_LOCAL_GPU = "llm_local_gpu"
    LLM_LOCAL_CPU = "llm_local_cpu"
    LLM_REMOTE_CLIENT = "llm_remote_client"
    INDEXER = "indexer"
    BUILD_TOOL = "build_tool"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    name: str
    cmdline: str
    category: ProcessCategory
    confidence: float  # 0..1


@dataclass
class GpuSnapshot:
    available: bool
    name: str | None = None
    utilization_pct: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    error: str | None = None


@dataclass
class SystemSnapshot:
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    gpu: GpuSnapshot
    processes: list[ProcessInfo] = field(default_factory=list)


@dataclass
class PolicyAction:
    kind: str
    pid: int
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyResult:
    actions: list[PolicyAction]
    notes: list[str] = field(default_factory=list)
