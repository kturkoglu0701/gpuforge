from gpuforge.actions import apply_action
from gpuforge.models import PolicyAction


def test_oom_and_affinity_dry_run() -> None:
    oom = apply_action(
        PolicyAction("set_oom_score_adj", 1, "x", {"value": 200}),
        dry_run=True,
    )
    assert oom.dry_run and oom.applied is False
    assert "oom_score_adj" in oom.message
    aff = apply_action(
        PolicyAction("set_cpu_affinity", 1, "x", {"cpus": [0]}),
        dry_run=True,
    )
    assert aff.result["cpus"] == [0]


def test_unknown_action_kind() -> None:
    out = apply_action(PolicyAction("noop", 1, "x", {}), dry_run=False)
    assert out.applied is False
    assert "unknown" in out.message
