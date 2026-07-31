"""YAML experiment config loading with recursive `extends` inheritance.

A config may extend a parent that itself extends another, and each level is deep-merged onto
the one above. That chaining is relied on: the Colab configs extend `resnet50_baseline.yaml`,
which extends `base.yaml`, to override only `data.num_workers` for a 2-vCPU runtime.
"""

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
    """Load a YAML config, deep-merging its `extends` chain (resolved relative to configs/).

    Deep-merge, not replace: a child overriding `train.lr` keeps the rest of the parent's
    `train` block. A shallow merge would silently drop optimizer, scheduler and early-stopping
    settings — training would still run, just not with the settings you think.
    """
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
