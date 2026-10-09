from pathlib import Path

from gpuforge.config import bundled_config_text, load_config


def test_bundled_config_has_optimizer() -> None:
    text = bundled_config_text()
    assert "optimizer:" in text
    assert "mode: adaptive" in text


def test_load_config_includes_thresholds(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        "thresholds:\n  gpu_util_high: 42\noptimizer:\n  mode: fixed\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg["thresholds"]["gpu_util_high"] == 42
    assert cfg["optimizer"]["mode"] == "fixed"
