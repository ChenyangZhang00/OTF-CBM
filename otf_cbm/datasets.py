from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .config import project_path


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(
    image_size: int,
    training: bool,
    resize_mode: str = "crop",
    augment: bool = True,
) -> Callable:
    tensor_first = False
    if resize_mode == "crop":
        operations: list[Callable] = [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
        ]
    elif resize_mode == "stretch":
        operations = [transforms.Resize((image_size, image_size))]
    elif resize_mode == "tensor_stretch":
        tensor_first = True
        operations = [
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
        ]
    else:
        raise ValueError(
            f"Unsupported resize_mode {resize_mode!r}; choose 'crop', "
            "'stretch', or 'tensor_stretch'."
        )
    if training and augment:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                ),
            ]
        )
    if not tensor_first:
        operations.append(transforms.ToTensor())
    operations.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(operations)


def build_transform_from_config(config: dict, training: bool) -> Callable:
    dataset_config = config.get("dataset", {})
    return build_transform(
        image_size=int(config.get("image_size", 224)),
        training=training,
        resize_mode=str(dataset_config.get("resize_mode", "crop")),
        augment=bool(dataset_config.get("train_augmentation", True)),
    )


class CUBDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        transform: Callable,
        crop_to_bbox: bool = True,
    ) -> None:
        self.root = root
        self.transform = transform
        images = _read_id_mapping(root / "images.txt", str)
        labels = _read_id_mapping(root / "image_class_labels.txt", int)
        split_flags = _read_id_mapping(root / "train_test_split.txt", int)
        class_mapping = _read_id_mapping(root / "classes.txt", str)
        boxes = (
            _read_bounding_boxes(root / "bounding_boxes.txt") if crop_to_bbox else {}
        )
        self.classes = [class_mapping[index] for index in sorted(class_mapping)]
        want_train = split == "train"
        self.samples = [
            (
                root / "images" / images[index],
                labels[index] - 1,
                boxes[index] if crop_to_bbox else None,
            )
            for index in sorted(images)
            if bool(split_flags[index]) == want_train
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label, box = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if box is not None:
                image = image.crop(box)
            tensor = self.transform(image)
        return tensor, label


def _read_id_mapping(path: Path, value_type: type) -> dict[int, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset annotation not found: {path}")
    mapping = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            identifier, value = line.rstrip("\n").split(maxsplit=1)
            mapping[int(identifier)] = value_type(value)
    return mapping


def _read_bounding_boxes(
    path: Path,
) -> dict[int, tuple[int, int, int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CUB bounding boxes not found: {path}")
    boxes = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            identifier, x, y, width, height = line.split()
            left = int(float(x))
            top = int(float(y))
            boxes[int(identifier)] = (
                left,
                top,
                int(float(x) + float(width)),
                int(float(y) + float(height)),
            )
    return boxes


class DatasetView(Dataset):
    def __init__(
        self, dataset: Dataset, indices: list[int], classes: list[str]
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.classes = classes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def _build_awa2(
    root: Path,
    split: str,
    transform: Callable,
    val_fraction: float,
    seed: int,
) -> DatasetView:
    image_root = root / "JPEGImages"
    base = datasets.ImageFolder(image_root, transform=transform)
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {index: [] for index in range(len(base.classes))}
    for index, (_, label) in enumerate(base.samples):
        by_class[label].append(index)
    train_indices, val_indices = [], []
    for indices in by_class.values():
        rng.shuffle(indices)
        val_count = max(1, round(len(indices) * val_fraction))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    selected = train_indices if split == "train" else val_indices
    return DatasetView(base, selected, list(base.classes))


def build_dataset(config: dict, split: str) -> Dataset:
    dataset_config = config["dataset"]
    dataset_type = dataset_config["type"].lower()
    root = project_path(config, dataset_config["root"])
    assert root is not None
    transform = build_transform_from_config(config, training=split == "train")

    if dataset_type == "cub":
        dataset = CUBDataset(
            root,
            split,
            transform,
            crop_to_bbox=bool(dataset_config.get("crop_to_bbox", True)),
        )
    elif dataset_type == "cifar100":
        dataset = datasets.CIFAR100(
            root=root,
            train=split == "train",
            transform=transform,
            download=False,
        )
    elif dataset_type in {"imagenet", "imagefolder"}:
        directory = dataset_config.get(
            f"{split}_directory", "train" if split == "train" else "val"
        )
        dataset = datasets.ImageFolder(root / directory, transform=transform)
    elif dataset_type == "places365":
        places_split = "train-standard" if split == "train" else "val"
        dataset = datasets.Places365(
            root=root,
            split=places_split,
            small=bool(dataset_config.get("small", True)),
            download=False,
            transform=transform,
        )
    elif dataset_type == "awa2":
        dataset = _build_awa2(
            root,
            split,
            transform,
            val_fraction=float(dataset_config.get("val_fraction", 0.2)),
            seed=int(config.get("seed", 1111)),
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    expected = dataset_config.get("num_classes")
    actual = len(getattr(dataset, "classes", []))
    if expected is not None and actual and int(expected) != actual:
        raise ValueError(
            f"{dataset_config['name']} has {actual} classes, expected {expected}. "
            "Check the dataset directory and class ordering."
        )
    return dataset
