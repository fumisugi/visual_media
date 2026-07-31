from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def category_names(config: dict[str, Any]) -> list[str]:
    return list(config["categories"].keys())


def prompt_set(config: dict[str, Any], prompt_index: int) -> list[str]:
    return [
        details["prompts"][prompt_index]
        for details in config["categories"].values()
    ]
