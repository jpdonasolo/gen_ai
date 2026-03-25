"""Load training configs from YAML and merge optional CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_train_yaml(path: str | Path) -> dict[str, Any]:
    """Load a training YAML file. Returns a plain dict (nested structures preserved)."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data
