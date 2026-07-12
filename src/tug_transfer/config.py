from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    config["_config_path"] = str(config_path.resolve())
    return config


def require(config: dict[str, Any], dotted_key: str) -> Any:
    """Read a required nested value using dot notation."""
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing required configuration key: {dotted_key}")
        value = value[part]
    return value


def resolve_path(value: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    """Resolve a path, optionally relative to a base directory."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path.resolve()
