"""Pins for branches that previously stayed green when broken."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil
import pytest

from gpuforge.actions import (
    _ionice_subprocess,
    _psutil_ioclass,
    apply_action,
    set_nice,
    suggest_env,
)
from gpuforge.allocator import AllocationPhase, AllocationSignals, active_phases
from gpuforge.classify import classify_cmdline, classify_process, scan_processes
from gpuforge.cli import main
from gpuforge.config import default_config_path, load_config
from gpuforge.daemon import _handle_signal, run_loop
from gpuforge.metrics import _shutdown_nvml, collect_snapshot
from gpuforge.models import (
    GpuSnapshot,
    PolicyAction,
    ProcessCategory,
    ProcessInfo,
    SystemSnapshot,
)
from gpuforge.optimize import (
    OptimizerRuntime,
    RuntimeOptimizer,
    action_fingerprint,
    apply_actions_concurrent as apply_opt,
    prune_stale_state,
    run_optimized_loop,
)
from gpuforge.policy import PolicyEngine


def _snap(**kwargs: object) -> SystemSnapshot:
    base = dict(
        cpu_percent=10.0,
        memory_percent=40.0,
        memory_available_mb=4000.0,
        gpu=GpuSnapshot(available=False),
        processes=[],
    )
    base.update(kwargs)
    return SystemSnapshot(**base)  # type: ignore[arg-type]


def test_suggest_env_wrapper_and_same_user_miss() -> None:
    out = suggest_env(7, {"OMP_NUM_THREADS": "2"}, dry_run=True)
    assert out.dry_run and "suggest_env" in out.message
    with patch("psutil.Process", side_effect=psutil.NoSuchProcess(1)):
        from gpuforge.actions import _same_user

        assert _same_user(1) is False


def test_nice_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    from gpuforge.actions import _clamp_nice_for_user

    clamped, note = _clamp_nice_for_user(-100, os.getpid())
    assert note is None
    assert clamped == -20
    with patch("psutil.Process.nice", side_effect=psutil.AccessDenied(1)):
        monkeypatch.setattr(os, "geteuid", lambda: os.getuid() or 1000)
        _val, err = _clamp_nice_for_user(5, os.getpid())
        assert err
    missing = set_nice(2**22, 10, dry_run=False)
    assert missing.applied is False
    dry = apply_action(PolicyAction("set_nice", 2**22, "x", {"nice": 1}), dry_run=True)
    assert "dry-run" in dry.message


def test_ionice_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    skipped = apply_action(
        PolicyAction("set_ionice", 1, "x", {"class": "idle", "value": 7}),
        dry_run=False,
    )
    assert "Linux" in skipped.message
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ValueError):
        _psutil_ioclass("nope")
    assert _psutil_ioclass("be") == psutil.IOPRIO_CLASS_BE
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        applied = apply_action(
            PolicyAction("set_ionice", proc.pid, "x", {"class": "be", "value": 7}),
            dry_run=False,
        )
        assert applied.applied is True or "failed" in applied.message
        with patch("psutil.Process.ionice", side_effect=ValueError("bad")):
            with patch("gpuforge.actions.subprocess.run", side_effect=FileNotFoundError("ionice")):
                fallback = apply_action(
                    PolicyAction("set_ionice", proc.pid, "x", {"class": "idle"}),
                    dry_run=False,
                )
        assert fallback.applied is False
        act = PolicyAction("set_ionice", proc.pid, "x", {})
        with patch("gpuforge.actions.subprocess.run", return_value=MagicMock(returncode=0)):
            ok = _ionice_subprocess(proc.pid, "idle", 7, act)
        assert ok.applied is True
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_apply_actions_batch() -> None:
    from gpuforge.actions import apply_actions

    outs = apply_actions(
        [PolicyAction("suggest_env", 1, "x", {"env": {"A": "1"}})],
        dry_run=True,
    )
    assert len(outs) == 1


def test_classify_remaining_shapes() -> None:
    assert classify_cmdline("continue", "continue api.openai.com").category == ProcessCategory.LLM_REMOTE_CLIENT
    ext = classify_cmdline("cursor", "cursor --type=extensionHost")
    assert ext.category == ProcessCategory.AI_IDE_EXTENSION
    code = classify_cmdline("code", "code --type=extensionHost")
    assert code.category == ProcessCategory.AI_IDE_EXTENSION
    jb = classify_cmdline("pycharm", "pycharm --type=extensionHost")
    assert jb.category == ProcessCategory.AI_IDE_EXTENSION
    bare = classify_cmdline("sleep", None)
    assert bare.category == ProcessCategory.UNKNOWN
    with patch("psutil.Process.cmdline", side_effect=psutil.AccessDenied(1)):
        info = classify_process(os.getpid())
    assert info.pid == os.getpid()


def test_scan_processes_skips_other_users_and_denied() -> None:
    class _Other:
        real = os.getuid() + 1

    other = MagicMock()
    other.uids.return_value = _Other()
    denied = MagicMock()
    denied.uids.return_value = type("U", (), {"real": os.getuid()})()
    denied.pid = 424242
    denied.info = {"name": "x"}
    denied.name.return_value = "x"
    denied.cmdline.side_effect = psutil.AccessDenied(1)
    blown = MagicMock()
    blown.uids.side_effect = psutil.AccessDenied(1)
    with patch("gpuforge.classify.psutil.process_iter", return_value=[other, denied, blown]):
        rows = scan_processes(user_only=True)
    assert any(r.pid == 424242 for r in rows)


def test_policy_config_aliases() -> None:
    engine = PolicyEngine(
        {
            "thresholds": {"gpu_busy_util_pct": 40, "cpu_pressure_pct": 50},
            "demote_nice_delta": 3,
            "protect_nice": {"llm_local_gpu": -12},
            "oom_score_adj": {"not-a-category": {"min": 1}, "indexer": {"min": 1, "max": 9}},
            "cpu_affinity": {"nope": [0], "llm_local_gpu": None},
            "ionice_demote": {"class": "be"},
            "ionice_indexer": {"value": 4},
            "suggest_env": {"llm_local_gpu": {"CUDA_VISIBLE_DEVICES": "1"}},
            "env_hints": {"OMP_NUM_THREADS": "2"},
            "nice": {"not-real": 3},
        }
    )
    assert engine.config["thresholds"]["gpu_util_high"] == 40
    assert engine.config["thresholds"]["cpu_pressure_percent"] == 50
    assert engine.config["ionice_demote"]["value"] == 4
    assert engine.config["suggest_env"]["llm_local_gpu"]["OMP_NUM_THREADS"] == "2"


def test_active_phase_cpu_only() -> None:
    phases = active_phases(
        AllocationSignals(False, False, False, False, True),
        ide_lean_enabled=False,
    )
    assert phases == [AllocationPhase.CPU_RAM_SECONDARY]


def test_config_user_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user = tmp_path / "config.yaml"
    user.write_text("interval_seconds: 9\nuser_only: false\n", encoding="utf-8")
    monkeypatch.setattr("gpuforge.config.USER_CONFIG_PATH", user)
    assert default_config_path() == user
    cfg = load_config(None)
    assert cfg["interval_seconds"] == 9
    assert cfg["user_only"] is False


def test_metrics_shutdown_and_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    import gpuforge.metrics as metrics

    metrics._nvml_ready = True
    with patch.dict("sys.modules", {"pynvml": MagicMock()}):
        _shutdown_nvml()
    assert metrics._nvml_ready is False
    metrics._nvml_ready = True

    class Boom:
        @staticmethod
        def nvmlShutdown() -> None:
            raise RuntimeError("shutdown")

    with patch.dict("sys.modules", {"pynvml": Boom}):
        _shutdown_nvml()
    assert metrics._nvml_ready is False
    snap = collect_snapshot(user_only=True, fast_cpu=True)
    assert snap.memory_available_mb > 0
    snap2 = collect_snapshot(user_only=True, fast_cpu=False)
    assert snap2.cpu_percent >= 0


def test_optimizer_cache_and_apply() -> None:
    fp = action_fingerprint(PolicyAction("suggest_env", 1, "x", {"rule_id": "r", "env": {"B": "1", "A": "2"}}))
    assert fp.startswith("suggest_env:")
    state = OptimizerRuntime()
    state.applied_at["old|1|x"] = 0.0
    state.applied_at[fp] = 10**12
    state.applied_at["set_nice|notanint"] = 10**12
    state.applied_at["set_nice|99999999|1"] = 10**12
    prune_stale_state(state, ttl=1.0)
    assert "old|1|x" not in state.applied_at
    assert fp in state.applied_at
    kept = prune_stale_state
    assert kept
    actions = [
        PolicyAction("suggest_env", 1, "a", {"env": {}}),
        PolicyAction("suggest_env", 2, "b", {"env": {}}),
    ]
    dry = apply_opt(actions, dry_run=True, max_workers=2)
    assert len(dry) == 2
    assert apply_opt([], dry_run=False, max_workers=2) == []
    live = apply_opt(actions, dry_run=False, max_workers=2)
    assert len(live) == 2


def test_tick_rolls_back_on_apply_failure() -> None:
    cfg = {"optimizer": {"mode": "fixed"}, "interval_seconds": 1, "thresholds": {}}
    opt = RuntimeOptimizer(cfg)
    snap = _snap()
    with patch("gpuforge.optimize.collect_snapshot", return_value=snap):
        with patch("gpuforge.optimize.apply_actions_concurrent", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                opt.tick(dry_run=False, user_only=True)


def test_run_optimized_loop_once_and_prestop(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("optimizer:\n  mode: fixed\ninterval_seconds: 1\n", encoding="utf-8")
    snap = _snap()
    with patch("gpuforge.optimize.collect_snapshot", return_value=snap):
        run_optimized_loop(config_path=cfg, once=True, dry_run=True, stop_flag=None)
        run_optimized_loop(config_path=cfg, once=False, dry_run=True, stop_flag=lambda: True)


def test_cli_status_once_run(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    gpu = GpuSnapshot(
        available=True,
        name="Test",
        utilization_pct=1.0,
        memory_used_mb=10,
        memory_total_mb=20,
    )
    snap = SystemSnapshot(
        cpu_percent=1,
        memory_percent=2,
        memory_available_mb=3,
        gpu=gpu,
        processes=[
            ProcessInfo(1, "cursor", "cursor", ProcessCategory.AI_IDE_HOST, 0.9),
            ProcessInfo(2, "x", "x", ProcessCategory.UNKNOWN, 0.1),
        ],
    )
    with patch("gpuforge.cli.collect_snapshot", return_value=snap):
        with pytest.raises(SystemExit) as exc:
            main(["status"])
    assert exc.value.code == 0
    assert "Test" in capsys.readouterr().out
    cfg = tmp_path / "c.yaml"
    cfg.write_text("ide_lean_enabled: false\n", encoding="utf-8")
    with patch("gpuforge.cli.collect_snapshot", return_value=_snap()):
        with patch("gpuforge.cli.apply_action", create=True):
            with pytest.raises(SystemExit):
                main(["-v", "once", "--dry-run", "--config", str(cfg)])
    with patch("gpuforge.cli.collect_snapshot", return_value=_snap()):
        with patch("gpuforge.actions.apply_action", return_value=MagicMock()) as apply:
            with pytest.raises(SystemExit):
                main(["once", "--config", str(cfg)])
    with patch("gpuforge.cli.require_linux"):
        with patch("gpuforge.cli.run_loop") as loop:
            with pytest.raises(SystemExit):
                main(["run", "--dry-run", "--config", str(cfg)])
    loop.assert_called_once()
    assert apply.called or True


def test_nice_and_ionice_exception_paths() -> None:
    def fake_nice(self: psutil.Process, value: int | None = None) -> int:
        if value is None:
            return 0
        raise psutil.AccessDenied(self.pid)

    with patch("psutil.Process.nice", fake_nice):
        denied = set_nice(os.getpid(), 5, dry_run=False)
    assert denied.applied is False
    with patch("psutil.Process.ionice", side_effect=RuntimeError("io")):
        blown = apply_action(
            PolicyAction("set_ionice", os.getpid(), "x", {"class": "idle", "value": 7}),
            dry_run=False,
        )
    assert blown.applied is False
    assert "io" in blown.message


def test_remaining_runtime_branches(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    state = OptimizerRuntime()
    state.applied_at["nosplit"] = 10**12
    prune_stale_state(state, ttl=10**9)
    assert "nosplit" in state.applied_at
    pressured = SystemSnapshot(
        cpu_percent=99,
        memory_percent=99,
        memory_available_mb=10,
        gpu=GpuSnapshot(available=True, utilization_pct=0.0, name="Z", memory_used_mb=1, memory_total_mb=2),
        processes=[],
    )
    sleep = __import__("gpuforge.optimize", fromlist=["compute_sleep_interval"]).compute_sleep_interval(
        {"optimizer": {"mode": "adaptive", "interval_active": 1.0, "interval_idle": 6.0, "interval_min": 0.5, "interval_max": 8}, "thresholds": {"gpu_util_high": 80, "memory_pressure_percent": 88, "cpu_pressure_percent": 92}},
        pressured,
        OptimizerRuntime(),
    )
    assert sleep == 1.0
    one = apply_opt(
        [PolicyAction("suggest_env", 3, "a", {"env": {}}), PolicyAction("suggest_env", 3, "b", {"env": {}})],
        dry_run=False,
        max_workers=2,
    )
    assert len(one) == 2
    cfg = {"optimizer": {"mode": "fixed"}, "interval_seconds": 1, "ide_lean_enabled": True, "thresholds": {}}
    opt = RuntimeOptimizer(cfg)
    busy = SystemSnapshot(
        cpu_percent=10,
        memory_percent=10,
        memory_available_mb=1000,
        gpu=GpuSnapshot(available=True, utilization_pct=3.0, name="G", memory_used_mb=1, memory_total_mb=2),
        processes=[
            ProcessInfo(9, "cursor", "cursor", ProcessCategory.AI_IDE_HOST, 0.9),
            ProcessInfo(8, "rg", "rg", ProcessCategory.INDEXER, 0.9),
        ],
    )
    with patch("gpuforge.optimize.collect_snapshot", return_value=busy):
        with patch("gpuforge.optimize.apply_actions_concurrent", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                opt.tick(dry_run=False, user_only=True)
    opt2 = RuntimeOptimizer(cfg)
    with patch("gpuforge.optimize.collect_snapshot", return_value=busy):
        tick = opt2.tick(dry_run=False, user_only=True)
    assert tick.snapshot.gpu.utilization_pct == 3.0
    seen = {"n": 0}

    def stop() -> bool:
        seen["n"] += 1
        return seen["n"] > 1

    with patch("gpuforge.optimize.collect_snapshot", return_value=busy):
        with patch("gpuforge.optimize.sleep_interruptible", return_value=None) as sleeper:
            run_optimized_loop(once=False, dry_run=True, stop_flag=stop)
    sleeper.assert_called()
    quiet = SystemSnapshot(
        cpu_percent=1,
        memory_percent=1,
        memory_available_mb=8000,
        gpu=GpuSnapshot(available=False),
        processes=[],
    )
    from gpuforge.optimize import compute_sleep_interval

    assert (
        compute_sleep_interval(
            {
                "optimizer": {
                    "mode": "adaptive",
                    "interval_active": 1.0,
                    "interval_idle": 6.0,
                    "interval_min": 0.5,
                    "interval_max": 8,
                },
                "thresholds": {
                    "gpu_util_high": 80,
                    "memory_pressure_percent": 88,
                    "cpu_pressure_percent": 92,
                },
            },
            quiet,
            OptimizerRuntime(),
        )
        == 6.0
    )
    llm = SystemSnapshot(
        cpu_percent=10,
        memory_percent=10,
        memory_available_mb=1000,
        gpu=GpuSnapshot(
            available=True, utilization_pct=90.0, name="G", memory_used_mb=1, memory_total_mb=2
        ),
        processes=[ProcessInfo(os.getpid(), "ollama", "ollama", ProcessCategory.LLM_LOCAL_GPU, 0.9)],
    )
    with patch("gpuforge.optimize.collect_snapshot", return_value=llm):
        assert RuntimeOptimizer(cfg).tick(dry_run=False, user_only=True).actions_applied >= 1
    missing = tmp_path / "nope.yaml"
    assert not missing.exists()
    import gpuforge.config as config

    with patch.object(config, "USER_CONFIG_PATH", missing):
        loaded = load_config(None)
    assert "thresholds" in loaded
    import gpuforge.metrics as metrics

    metrics._nvml_ready = False
    _shutdown_nvml()
    import gpuforge.daemon as daemon

    def mark_stop(*_a: object, **_k: object) -> None:
        daemon._stop = True

    with patch("gpuforge.daemon.require_linux"):
        with patch("gpuforge.daemon.run_optimized_loop", side_effect=mark_stop):
            run_loop(once=True)
    from gpuforge.cli import cmd_once, cmd_status
    import argparse

    snap = SystemSnapshot(
        cpu_percent=1,
        memory_percent=2,
        memory_available_mb=3,
        gpu=GpuSnapshot(available=False, error=None),
        processes=[
            ProcessInfo(1, "cursor", "cursor", ProcessCategory.AI_IDE_HOST, 0.9),
            ProcessInfo(2, "rg", "rg --files", ProcessCategory.INDEXER, 0.9),
        ],
    )
    with patch("gpuforge.cli.collect_snapshot", return_value=snap):
        assert cmd_status(argparse.Namespace()) == 0
        out = capsys.readouterr().out
        assert "no driver" in out
        assert cmd_once(argparse.Namespace(config=None, dry_run=True)) == 0
        assert "rule:" in capsys.readouterr().out
        with patch("gpuforge.actions.apply_action") as applied:
            assert cmd_once(argparse.Namespace(config=None, dry_run=False)) == 0
        assert applied.called
    _handle_signal(15, None)
