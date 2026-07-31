"""YAML experiment config loading with single-level `extends` inheritance."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, merging its `extends` parent (resolved relative to configs/)."""
    path = Path(path)
    if not path.exists() and (CONFIG_DIR / path).exists():
        path = CONFIG_DIR / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent_name = cfg.pop("extends", None)
    if parent_name:
        parent = load_config(path.parent / parent_name)
        cfg = _deep_merge(parent, cfg)

    # Environment overrides
    data_root = os.environ.get("CROPGUARD_DATA_DIR")
    if data_root:
        cfg.setdefault("data", {})["root"] = data_root
    return cfg


def data_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["data"]["root"]).expanduser().resolve()
