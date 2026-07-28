from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .datasets import build_transform
from .models.cost import CostModel
from .transport import unbalanced_sinkhorn_log


def resolve_paco_image(image_root: Path, value: str | int) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() and relative.is_file():
        return relative

    candidates = [
        image_root / relative,
        image_root / relative.name,
        image_root / "train2017" / relative.name,
        image_root / "val2017" / relative.name,
    ]
    if relative.parts and relative.parts[0].lower() == image_root.name.lower():
        candidates.append(image_root.joinpath(*relative.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"PACO image {value!r} was not found under {image_root}")


class PACOImageLevelDataset(Dataset):
    """Prepared PACO image records consumed by Stage 1."""

    def __init__(
        self,
        json_path: Path,
        image_root: Path,
        image_size: int,
        resize_mode: str = "stretch",
    ) -> None:
        with json_path.open("r", encoding="utf-8") as handle:
            self.records = json.load(handle)
        self.image_root = image_root
        self.transform = build_transform(
            image_size,
            training=False,
            resize_mode=resize_mode,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        value = record.get("file_name") or record.get("image_id")
        if value is None:
            raise KeyError("PACO record has neither file_name nor image_id")
        path = resolve_paco_image(self.image_root, value)
        with Image.open(path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        components = record["components"]
        return {
            "image": image_tensor,
            "texts": [component["text"] for component in components],
            "patch_ids": [component["mask_patch_ids"] for component in components],
        }


def build_ground_truth_plan(
    patch_ids: list[list[int]],
    num_patches: int,
    device: torch.device,
) -> torch.Tensor:
    plan = torch.zeros(num_patches, len(patch_ids), device=device)
    for concept_index, identifiers in enumerate(patch_ids):
        valid = sorted(
            {
                min(max(int(identifier), 0), num_patches - 1)
                for identifier in identifiers
            }
        )
        if valid:
            plan[valid, concept_index] = 1.0 / len(valid)
    return plan


def inverse_ot_loss(
    model: CostModel,
    patch_features: torch.Tensor,
    text_features: torch.Tensor,
    patch_ids: list[list[int]],
    epsilon: float,
    tau: float,
    iterations: int,
    l1_weight: float,
) -> torch.Tensor | None:
    ground_truth = build_ground_truth_plan(
        patch_ids, patch_features.shape[0], patch_features.device
    )
    if ground_truth.sum() == 0:
        return None
    source_mass = ground_truth.sum(dim=1)
    target_mass = ground_truth.sum(dim=0)
    cost, _ = model(patch_features, text_features)
    cost = cost / (cost.std() + 1e-6)
    predicted = unbalanced_sinkhorn_log(
        cost,
        source_mass,
        target_mass,
        epsilon=epsilon,
        tau=tau,
        iterations=iterations,
    )
    reconstruction = torch.abs((predicted - ground_truth) * cost).sum()
    return reconstruction + l1_weight * model.theta.abs().sum()
