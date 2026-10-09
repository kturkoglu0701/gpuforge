from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from gpuforge.policy import _normalize_config

USER_CONFIG_PATH = Path.home() / ".config" / "gpuforge" / "config.yaml"


def bundled_config_text() -> str:
    return resources.files("gpuforge").joinpath("data/default.yaml").read_text(encoding="utf-8")


def default_config_path() -> Path:
    """Prefer user override; else bundled path marker (load_config reads bundle)."""
    if USER_CONFIG_PATH.is_file():
        return USER_CONFIG_PATH
    return Path("__bundled__")


DEFAULT_CONFIG_PATH = default_config_path()


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load YAML: explicit path, ~/.config/gpuforge/config.yaml, or bundled package defaults."""
    runtime_keys = ("interval_seconds", "dry_run", "user_only", "ide_lean_enabled")
    runtime: dict[str, Any] = {
        "interval_seconds": 5,
        "dry_run": False,
        "user_only": True,
        "ide_lean_enabled": True,
    }

    if path is not None:
        raw_text = path.read_text(encoding="utf-8")
    elif USER_CONFIG_PATH.is_file():
        raw_text = USER_CONFIG_PATH.read_text(encoding="utf-8")
    else:
        raw_text = bundled_config_text()

    file_data = yaml.safe_load(raw_text) or {}
    for key in runtime_keys:
        if key in file_data:
            runtime[key] = file_data[key]

    policy_raw = {k: v for k, v in file_data.items() if k not in runtime_keys}
    policy = _normalize_config(policy_raw)
    return {**runtime, **policy_raw, **policy}
