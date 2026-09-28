'YAML config loading with dotted-path CLI overrides.'
from __future__ import annotations

import yaml


def get_by_path(cfg: dict, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_by_path(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot override {dotted}: {part} is not a mapping")
    node[parts[-1]] = value


def load_config(path: str, overrides=()) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for ov in overrides or ():
        key, sep, val = ov.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got: {ov}")
        set_by_path(cfg, key.strip(), yaml.safe_load(val))
    return cfg
