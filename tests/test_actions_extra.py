from gpuforge.actions import apply_action
from gpuforge.models import PolicyAction


def test_unknown_action_kind() -> None:
    out = apply_action(PolicyAction("noop", 1, "x", {}), dry_run=False)
    assert out.applied is False
    assert "unknown" in out.message
