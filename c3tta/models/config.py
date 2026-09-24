from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class Config:
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    tta: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        parent = raw.pop("extends", None)
        if parent:
            base = cls.from_yaml(path.parent / parent)
            raw = _deep_merge(base.__dict__, raw)
        return cls(**{key: raw.get(key, {}) for key in ("data", "model", "train", "tta", "hardware")})


def load_config(path: str | Path) -> Config:
    return Config.from_yaml(path)


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
