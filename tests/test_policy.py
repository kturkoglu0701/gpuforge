from __future__ import annotations

from gpuforge.models import GpuSnapshot, ProcessCategory, ProcessInfo, SystemSnapshot
from gpuforge.policy import RULE_IDS, PolicyEngine


def _proc(pid: int, cat: ProcessCategory) -> ProcessInfo:
    return ProcessInfo(pid=pid, name="x", cmdline="x", category=cat, confidence=0.9)


def _snapshot(
    *,
    mem_pct: float = 50.0,
    cpu_pct: float = 10.0,
    gpu_util: float | None = 50.0,
    processes: list[ProcessInfo] | None = None,
) -> SystemSnapshot:
    return SystemSnapshot(
        cpu_percent=cpu_pct,
        memory_percent=mem_pct,
        memory_available_mb=8000.0,
        gpu=GpuSnapshot(available=True, utilization_pct=gpu_util),
        processes=processes or [],
    )


def test_rules_summary_lists_three_rules() -> None:
    engine = PolicyEngine({"thresholds": {"gpu_util_high": 90, "memory_pressure_percent": 88}})
    summary = engine.rules_summary()
    assert len(summary) == 3
    assert summary[0]["id"] == RULE_IDS[0]


def test_ide_session_demotes_indexer_without_llm() -> None:
    engine = PolicyEngine()
    snap = _snapshot(
        processes=[
            _proc(10, ProcessCategory.AI_IDE_HOST),
            _proc(200, ProcessCategory.INDEXER),
        ]
    )
    result = engine.evaluate(snap)
    assert RULE_IDS[1] in result.notes
    assert any(a.pid == 200 and a.kind == "set_nice" for a in result.actions)


def test_llm_active_demotes_indexer_via_gpu_rule() -> None:
    engine = PolicyEngine()
    snap = _snapshot(
        processes=[
            _proc(100, ProcessCategory.LLM_LOCAL_GPU),
            _proc(200, ProcessCategory.INDEXER),
            _proc(300, ProcessCategory.BUILD_TOOL),
        ]
    )
    result = engine.evaluate(snap)
    assert RULE_IDS[0] in result.notes
    kinds = {(a.pid, a.kind) for a in result.actions}
    assert (200, "set_nice") in kinds
    assert (300, "set_nice") in kinds
    assert (200, "set_ionice") in kinds
    nice_200 = next(a for a in result.actions if a.pid == 200 and a.kind == "set_nice")
    assert nice_200.payload["nice"] == 10 + 8
    oom_200 = next(a for a in result.actions if a.pid == 200 and a.kind == "set_oom_score_adj")
    assert oom_200.payload["value"] == 400


def test_gpu_high_protects_llm() -> None:
    engine = PolicyEngine()
    snap = _snapshot(gpu_util=95.0, processes=[_proc(42, ProcessCategory.LLM_LOCAL_GPU)])
    result = engine.evaluate(snap)
    assert RULE_IDS[0] in result.notes
    protect = [a for a in result.actions if a.pid == 42 and a.payload.get("rule_id") == RULE_IDS[0]]
    assert any(a.kind == "set_nice" and a.payload["nice"] == -15 for a in protect)
    assert any(a.kind == "suggest_env" for a in protect)
    oom = next(a for a in protect if a.kind == "set_oom_score_adj")
    assert oom.payload["value"] == -900


def test_ram_pressure_demotes_indexer_and_extension() -> None:
    engine = PolicyEngine()
    snap = _snapshot(
        mem_pct=92.0,
        processes=[
            _proc(55, ProcessCategory.AI_IDE_EXTENSION),
            _proc(56, ProcessCategory.INDEXER),
        ],
    )
    result = engine.evaluate(snap)
    assert RULE_IDS[2] in result.notes
    assert any(a.pid == 55 and a.kind == "set_nice" for a in result.actions)
    assert any(a.pid == 56 and a.kind == "set_nice" for a in result.actions)


def test_yaml_threshold_aliases() -> None:
    engine = PolicyEngine({"thresholds": {"memory_pressure_pct": 70}})
    snap = _snapshot(
        mem_pct=75.0,
        processes=[_proc(9, ProcessCategory.AI_IDE_EXTENSION)],
    )
    result = engine.evaluate(snap)
    assert RULE_IDS[2] in result.notes


def test_config_string_categories_and_optional_affinity() -> None:
    engine = PolicyEngine(
        {
            "nice": {"indexer": 20},
            "cpu_affinity": {"llm_local_gpu": [0, 1]},
            "thresholds": {"gpu_util_high": 70},
        }
    )
    snap = _snapshot(
        gpu_util=80.0,
        processes=[_proc(1, ProcessCategory.LLM_LOCAL_GPU)],
    )
    result = engine.evaluate(snap)
    affinity = [a for a in result.actions if a.kind == "set_cpu_affinity"]
    assert affinity and affinity[0].payload["cpus"] == [0, 1]
