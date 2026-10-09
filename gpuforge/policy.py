from __future__ import annotations

from copy import deepcopy
from typing import Any

from gpuforge.allocator import read_signals
from gpuforge.models import PolicyAction, PolicyResult, ProcessCategory, ProcessInfo, SystemSnapshot

DEFAULT_POLICY_CONFIG: dict[str, Any] = {
    "ide_lean_enabled": True,
    "thresholds": {
        "gpu_util_high": 55.0,
        "memory_pressure_percent": 88.0,
        "cpu_pressure_percent": 92.0,
    },
    "nice": {
        ProcessCategory.LLM_LOCAL_GPU: -10,
        ProcessCategory.LLM_LOCAL_CPU: 0,
        ProcessCategory.LLM_REMOTE_CLIENT: 2,
        ProcessCategory.AI_IDE_HOST: 5,
        ProcessCategory.AI_IDE_EXTENSION: 8,
        ProcessCategory.INDEXER: 10,
        ProcessCategory.BUILD_TOOL: 12,
        ProcessCategory.UNKNOWN: 0,
    },
    "demote_nice_delta": 8,
    "protect_nice": {
        ProcessCategory.LLM_LOCAL_GPU: -15,
    },
    "oom_score_adj": {
        ProcessCategory.LLM_LOCAL_GPU: {"min": -900, "max": -300},
        ProcessCategory.AI_IDE_EXTENSION: {"min": 100, "max": 500},
        ProcessCategory.INDEXER: {"min": 50, "max": 400},
        ProcessCategory.BUILD_TOOL: {"min": 50, "max": 400},
    },
    "cpu_affinity": {},
    "ionice_demote": {"class": "idle", "value": 7},
    "suggest_env": {
        "llm_local_gpu": {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "4",
        },
    },
}

RULE_IDS = (
    "ide_session_lean_cpu",
    "gpu_operational_protect_compute",
    "llm_active_demote_indexer_build",
    "cpu_ram_pressure_secondary",
)


class PolicyEngine:
    """Evaluate orchestration rules from a YAML-like config dict."""

    def __init__(self, config: dict[str, Any] | None = None, *, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self.config = _normalize_config(config or {})

    def rules_summary(self) -> list[dict[str, str]]:
        t = self.config["thresholds"]
        return [
            {
                "id": RULE_IDS[0],
                "when": "AI IDE host detected (Cursor, VS Code, …)",
                "then": "Proactively demote indexer/build_tool — keep CPU/RAM free for GPU",
            },
            {
                "id": RULE_IDS[1],
                "when": f"GPU util >= {t['gpu_util_high']}% or local GPU LLM process",
                "then": "Protect GPU compute; shed IDE extension CPU while GPU is busy",
            },
            {
                "id": RULE_IDS[2],
                "when": "llm_local_gpu process running",
                "then": "Demote indexer/build_tool competing with inference",
            },
            {
                "id": RULE_IDS[3],
                "when": f"RAM >= {t['memory_pressure_percent']}% or CPU >= {t['cpu_pressure_percent']}%",
                "then": "Secondary: demote IDE extensions and remote LLM clients",
            },
        ]

    def evaluate(self, snapshot: SystemSnapshot) -> PolicyResult:
        actions: list[PolicyAction] = []
        notes: list[str] = []
        by_cat = _group_by_category(snapshot.processes)
        t = self.config["thresholds"]
        signals = read_signals(
            snapshot,
            gpu_util_high=float(t["gpu_util_high"]),
            memory_pressure_percent=float(t["memory_pressure_percent"]),
            cpu_pressure_percent=float(t.get("cpu_pressure_percent", 92.0)),
        )
        if bool(self.config.get("ide_lean_enabled", True)) and signals.ide_active:
            notes.append(RULE_IDS[0])
            actions.extend(
                _demote_categories(
                    by_cat,
                    (ProcessCategory.INDEXER, ProcessCategory.BUILD_TOOL),
                    self.config,
                    rule_id=RULE_IDS[0],
                )
            )

        gpu_phase = signals.llm_gpu_active or signals.gpu_busy
        if gpu_phase:
            notes.append(RULE_IDS[1])
            actions.extend(
                _protect_llm(
                    by_cat.get(ProcessCategory.LLM_LOCAL_GPU, []),
                    self.config,
                    rule_id=RULE_IDS[1],
                )
            )
            actions.extend(
                _demote_categories(
                    by_cat,
                    (ProcessCategory.AI_IDE_EXTENSION, ProcessCategory.INDEXER, ProcessCategory.BUILD_TOOL),
                    self.config,
                    rule_id=RULE_IDS[1],
                )
            )

        if signals.llm_gpu_active:
            notes.append(RULE_IDS[2])
            actions.extend(
                _demote_categories(
                    by_cat,
                    (ProcessCategory.INDEXER, ProcessCategory.BUILD_TOOL),
                    self.config,
                    rule_id=RULE_IDS[2],
                )
            )

        if signals.ram_pressure or signals.cpu_pressure:
            notes.append(RULE_IDS[3])
            actions.extend(
                _demote_categories(
                    by_cat,
                    (ProcessCategory.AI_IDE_EXTENSION, ProcessCategory.LLM_REMOTE_CLIENT),
                    self.config,
                    rule_id=RULE_IDS[3],
                )
            )

        return PolicyResult(actions=_dedupe_actions(actions), notes=notes)


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_POLICY_CONFIG)
    if "thresholds" in raw:
        thresholds = dict(raw["thresholds"])
        if "gpu_busy_util_pct" in thresholds and "gpu_util_high" not in thresholds:
            thresholds["gpu_util_high"] = thresholds["gpu_busy_util_pct"]
        if "memory_pressure_pct" in thresholds and "memory_pressure_percent" not in thresholds:
            thresholds["memory_pressure_percent"] = thresholds["memory_pressure_pct"]
        if "cpu_pressure_pct" in thresholds and "cpu_pressure_percent" not in thresholds:
            thresholds["cpu_pressure_percent"] = thresholds["cpu_pressure_pct"]
        cfg["thresholds"].update(thresholds)
    if "ide_lean_enabled" in raw:
        cfg["ide_lean_enabled"] = bool(raw["ide_lean_enabled"])
    if "nice" in raw:
        cfg["nice"].update(_category_keyed(raw["nice"]))
    if "demote_nice_delta" in raw:
        cfg["demote_nice_delta"] = int(raw["demote_nice_delta"])
    if "protect_nice" in raw:
        cfg["protect_nice"].update(_category_keyed(raw["protect_nice"]))
    if "oom_score_adj" in raw:
        for key, val in raw["oom_score_adj"].items():
            cat = _parse_category(key)
            if cat is not None:
                cfg["oom_score_adj"][cat] = dict(val)
    if "cpu_affinity" in raw:
        for key, val in raw["cpu_affinity"].items():
            cat = _parse_category(key)
            if cat is not None and val is not None:
                cfg["cpu_affinity"][cat] = list(val)
    if "ionice_demote" in raw:
        cfg["ionice_demote"].update(raw["ionice_demote"])
    if "ionice_indexer" in raw:
        ion = raw["ionice_indexer"]
        cfg["ionice_demote"] = {
            "class": "idle",
            "value": int(ion.get("value", 7)),
        }
    if "suggest_env" in raw:
        cfg["suggest_env"].update(raw["suggest_env"])
    if "env_hints" in raw:
        llm_env = cfg["suggest_env"].setdefault("llm_local_gpu", {})
        llm_env.update({str(k): str(v) for k, v in raw["env_hints"].items()})
    return cfg


def _category_keyed(table: dict[str, Any]) -> dict[ProcessCategory, Any]:
    out: dict[ProcessCategory, Any] = {}
    for key, val in table.items():
        cat = _parse_category(key)
        if cat is not None:
            out[cat] = val
    return out


def _parse_category(key: str) -> ProcessCategory | None:
    try:
        return ProcessCategory(key)
    except ValueError:
        return None


def _group_by_category(processes: list[ProcessInfo]) -> dict[ProcessCategory, list[ProcessInfo]]:
    out: dict[ProcessCategory, list[ProcessInfo]] = {}
    for proc in processes:
        out.setdefault(proc.category, []).append(proc)
    return out


def _base_nice(config: dict[str, Any], category: ProcessCategory) -> int:
    return int(config["nice"].get(category, config["nice"].get(ProcessCategory.UNKNOWN, 0)))


def _demote_categories(
    by_cat: dict[ProcessCategory, list[ProcessInfo]],
    categories: tuple[ProcessCategory, ...],
    config: dict[str, Any],
    *,
    rule_id: str,
) -> list[PolicyAction]:
    actions: list[PolicyAction] = []
    delta = int(config["demote_nice_delta"])
    ionice = config["ionice_demote"]
    for cat in categories:
        for proc in by_cat.get(cat, []):
            target_nice = _base_nice(config, cat) + delta
            actions.append(
                PolicyAction(
                    kind="set_nice",
                    pid=proc.pid,
                    detail=f"{rule_id}: demote {cat.value} pid={proc.pid}",
                    payload={"nice": target_nice, "rule_id": rule_id},
                )
            )
            actions.append(
                PolicyAction(
                    kind="set_ionice",
                    pid=proc.pid,
                    detail=f"{rule_id}: idle ionice for {cat.value}",
                    payload={
                        "class": ionice.get("class", "idle"),
                        "value": ionice.get("value", 7),
                        "rule_id": rule_id,
                    },
                )
            )
            oom = config["oom_score_adj"].get(cat)
            if oom:
                actions.append(
                    PolicyAction(
                        kind="set_oom_score_adj",
                        pid=proc.pid,
                        detail=f"{rule_id}: oom_score_adj for {cat.value}",
                        payload={
                            "value": int(oom.get("max", oom.get("min", 0))),
                            "rule_id": rule_id,
                        },
                    )
                )
    return actions


def _protect_llm(
    llm_procs: list[ProcessInfo],
    config: dict[str, Any],
    *,
    rule_id: str,
) -> list[PolicyAction]:
    actions: list[PolicyAction] = []
    protect_nice = int(
        config["protect_nice"].get(
            ProcessCategory.LLM_LOCAL_GPU,
            config["nice"].get(ProcessCategory.LLM_LOCAL_GPU, -10),
        )
    )
    oom = config["oom_score_adj"].get(ProcessCategory.LLM_LOCAL_GPU, {})
    oom_val = int(oom.get("min", -500))
    env_hints = config["suggest_env"].get("llm_local_gpu", {})
    affinity = config["cpu_affinity"].get(ProcessCategory.LLM_LOCAL_GPU)

    for proc in llm_procs:
        actions.append(
            PolicyAction(
                kind="set_nice",
                pid=proc.pid,
                detail=f"{rule_id}: protect llm_local_gpu pid={proc.pid}",
                payload={"nice": protect_nice, "rule_id": rule_id},
            )
        )
        if oom:
            actions.append(
                PolicyAction(
                    kind="set_oom_score_adj",
                    pid=proc.pid,
                    detail=f"{rule_id}: favor llm oom_score_adj",
                    payload={"value": oom_val, "rule_id": rule_id},
                )
            )
        if affinity:
            actions.append(
                PolicyAction(
                    kind="set_cpu_affinity",
                    pid=proc.pid,
                    detail=f"{rule_id}: pin llm CPU affinity",
                    payload={"cpus": list(affinity), "rule_id": rule_id},
                )
            )
        if env_hints:
            actions.append(
                PolicyAction(
                    kind="suggest_env",
                    pid=proc.pid,
                    detail=f"{rule_id}: launcher env for llm",
                    payload={"env": dict(env_hints), "rule_id": rule_id},
                )
            )
    return actions


def _dedupe_actions(actions: list[PolicyAction]) -> list[PolicyAction]:
    seen: set[tuple[str, int, str]] = set()
    out: list[PolicyAction] = []
    for act in actions:
        rule_id = str(act.payload.get("rule_id", ""))
        key = (act.kind, act.pid, rule_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(act)
    return out
