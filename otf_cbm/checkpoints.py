from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models.cost import CostModel
from .models.otf_cbm import OTFCBM


def _legacy_cost_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state.items():
        if key.startswith("iot_cost_model."):
            key = key[len("iot_cost_model.") :]
        if key.startswith("cost_model."):
            key = key[len("cost_model.") :]
        key = key.replace("img_proj.ln.", "img_proj.norm.")
        key = key.replace("img_proj.short.", "img_proj.shortcut.")
        converted[key] = value
    return converted


def load_stage1_cost(
    model: CostModel,
    path: str | Path,
) -> None:
    payload: Any = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Stage-1 checkpoint must be a dictionary")
    state = payload.get("cost_model", payload.get("model_state", payload))
    if not isinstance(state, dict):
        raise TypeError("No cost-model state dictionary in Stage-1 checkpoint")
    cost_state = _legacy_cost_keys(
        {key: value for key, value in state.items() if torch.is_tensor(value)}
    )
    compatible = {
        key: value for key, value in cost_state.items() if key in model.state_dict()
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    required_missing = [
        key for key in missing if key == "theta" or key.startswith("img_proj.")
    ]
    if required_missing:
        raise RuntimeError(
            f"Stage-1 checkpoint is missing required keys: {required_missing}"
        )
    if unexpected:
        print(f"Ignored {len(unexpected)} unexpected Stage-1 checkpoint keys")


def save_stage1_checkpoint(
    path: str | Path,
    model: CostModel,
    epoch: int,
    config: dict,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "stage": 1,
        "epoch": epoch,
        "cost_model": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "config": {
            key: value for key, value in config.items() if not key.startswith("_")
        },
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_stage2_checkpoint(
    path: str | Path,
    model: OTFCBM,
    epoch: int,
    config: dict,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "stage": 2,
        "epoch": epoch,
        "model_state": model.export_state_dict(),
        "num_classes": model.num_classes,
        "num_concepts": model.num_concepts,
        "class_names": model.concept_bank.class_names,
        "config": {
            key: value for key, value in config.items() if not key.startswith("_")
        },
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_stage2_checkpoint(
    model: OTFCBM,
    path: str | Path,
) -> dict[str, Any]:
    payload: Any = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise TypeError("Not a clean OTF-CBM Stage-2 checkpoint")
    if int(payload.get("num_classes", -1)) != model.num_classes:
        raise ValueError("Checkpoint and dataset class counts differ")
    if int(payload.get("num_concepts", -1)) != model.num_concepts:
        raise ValueError(
            "Checkpoint and concept-bank dimensions differ. Use the concept bank "
            "that was used for training."
        )
    checkpoint_classes = [str(name) for name in payload.get("class_names", [])]
    if checkpoint_classes != model.concept_bank.class_names:
        raise ValueError(
            "Checkpoint and concept-bank class ordering differ. "
            "Use the exact concept bank saved for this checkpoint."
        )
    incompatible = model.load_state_dict(payload["model_state"], strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("backbone.") and key != "global_concepts"
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return payload
