from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def get_cfg(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_cfg(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur: Any = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
        if not isinstance(cur, dict):
            raise ValueError(f"Cannot set {dotted_key}: {part} is not a mapping")
    cur[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        set_cfg(cfg, key, yaml.safe_load(raw_value))
    return cfg

