"""Small YAML config helper.

Configs are plain nested dicts. A model config is merged on top of configs/base.yaml,
and command-line overrides use dotted keys, e.g. ``--set lr=0.005 split.seed=1``.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_value(text: str) -> Any:
    # yaml handles ints, floats, bools, null and lists like [1,2]
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def resolve_model_config(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml"):
        return p if p.is_absolute() else PROJECT_ROOT / p
    return CONFIG_DIR / "models" / f"{name_or_path}.yaml"


def load_config(model: str | None = None, overrides: Iterable[str] | None = None,
                base_path: str | Path | None = None) -> dict:
    """Load base.yaml, optionally merge a model config and dotted overrides."""
    base = read_yaml(base_path or CONFIG_DIR / "base.yaml")
    if model is not None:
        base = deep_merge(base, {"model_cfg": read_yaml(resolve_model_config(model))})
    return apply_overrides(base, overrides)


def project_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p
