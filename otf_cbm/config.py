from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _parse_override(value: str) -> Any:
    return yaml.safe_load(value)


def _set_nested(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def apply_overrides(
    config: dict[str, Any], overrides: list[str] | None
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected key=value")
        key, value = item.split("=", 1)
        _set_nested(result, key, _parse_override(value))
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    config = apply_overrides(config, overrides)
    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent.parent)
    return config


def project_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def require_file(config: dict[str, Any], value: str | Path, label: str) -> Path:
    path = project_path(config, value)
    assert path is not None
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path
