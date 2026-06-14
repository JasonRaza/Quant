"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path, base: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or project_root()) / p
