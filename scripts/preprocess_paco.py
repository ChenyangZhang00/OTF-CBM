#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from otf_cbm.stage1 import resolve_paco_image


INVALID_ATTRIBUTES = (
    "other",
    "unknown",
    "misc",
    "none",
    "material",
    "pattern",
    "marking",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PACO annotations to Stage-1 image-level records"
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument(
        "--image-root",
        required=True,
        help="Root used to verify image file_name entries",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--minimum-coverage", type=float, default=0.01)
    return parser.parse_args()


def clean_category(name: str) -> str:
    if ":" in name:
        object_name, part_name = name.split(":", 1)
    else:
        object_name, part_name = name, None
    object_name = re.sub(r"_\([^)]*\)", "", object_name).replace("_", " ").strip()
    if part_name:
        return f"{object_name}'s {part_name.replace('_', ' ').strip()}"
    return object_name


def compose_text(category: str, attributes: list[str]) -> str:
    attributes = [
        attribute.lower().strip()
        for attribute in attributes
        if not any(token in attribute.lower() for token in INVALID_ATTRIBUTES)
    ]
    category = clean_category(category)
    if not attributes:
        return f"a {category}"
    if len(attributes) == 1:
        return f"a {attributes[0]} {category}"
    if len(attributes) == 2:
        return f"a {attributes[0]} and {attributes[1]} {category}"
    return f"a {', '.join(attributes[:-1])}, and {attributes[-1]} {category}"


def decode_mask(segmentation, height: int, width: int) -> np.ndarray | None:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise RuntimeError(
            "PACO preprocessing requires pycocotools; install requirements.txt"
        ) from error
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict) and "counts" in segmentation:
        rle = segmentation
    else:
        return None
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask.any(axis=2)
    return mask.astype(np.float32)


def mask_to_patch_ids(
    mask: np.ndarray,
    input_size: int,
    patch_size: int,
    minimum_coverage: float,
) -> list[int]:
    if input_size % patch_size:
        raise ValueError("input size must be divisible by patch size")
    if mask.shape[0] <= 0 or mask.shape[1] <= 0:
        raise ValueError("mask must have positive height and width")
    tensor = torch.from_numpy(mask)[None, None]
    resized = F.interpolate(
        tensor,
        size=(input_size, input_size),
        mode="nearest",
    )[0, 0]
    grid = input_size // patch_size
    coverage = (
        resized.reshape(grid, patch_size, grid, patch_size)
        .permute(0, 2, 1, 3)
        .mean(dim=(2, 3))
        .flatten()
    )
    selected = torch.where(coverage > minimum_coverage)[0].tolist()
    return selected or [int(coverage.argmax())]


def main() -> None:
    args = parse_args()
    annotation_path = Path(args.annotations).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    with annotation_path.open("r", encoding="utf-8") as handle:
        paco = json.load(handle)

    images = {image["id"]: image for image in paco["images"]}
    categories = {category["id"]: category for category in paco["categories"]}
    attributes = {
        attribute["id"]: attribute["name"] for attribute in paco.get("attributes", [])
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    skipped_missing_images = 0
    for annotation in tqdm(paco["annotations"], desc="PACO annotations"):
        category = categories.get(annotation["category_id"], {})
        if category.get("supercategory") != "PART":
            continue
        attribute_names = [
            attributes[identifier]
            for identifier in annotation.get("attribute_ids", [])
            if identifier in attributes
        ]
        if not attribute_names:
            continue
        image = images.get(annotation["image_id"])
        if image is None:
            continue
        relative_path = image["file_name"]
        try:
            resolved_image = resolve_paco_image(image_root, relative_path)
        except FileNotFoundError:
            skipped_missing_images += 1
            continue
        mask = decode_mask(
            annotation.get("segmentation"), image["height"], image["width"]
        )
        if mask is None:
            continue
        text = compose_text(category.get("name", "unknown"), attribute_names)
        patch_ids = mask_to_patch_ids(
            mask,
            input_size=args.input_size,
            patch_size=args.patch_size,
            minimum_coverage=args.minimum_coverage,
        )
        if not patch_ids:
            continue
        try:
            stored_path = str(resolved_image.relative_to(image_root))
        except ValueError:
            stored_path = str(resolved_image)
        grouped[stored_path].append({"text": text, "mask_patch_ids": patch_ids})

    records = [
        {"file_name": file_name, "components": components}
        for file_name, components in grouped.items()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)
    print(
        f"saved={output_path} images={len(records)} "
        f"missing_image_annotations={skipped_missing_images}"
    )


if __name__ == "__main__":
    main()
