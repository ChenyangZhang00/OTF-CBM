#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from otf_cbm.config import load_config
from otf_cbm.datasets import build_dataset
from otf_cbm.factory import build_stage2_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Stage-2 checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    requested_device = config.get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    model = build_stage2_model(config, device, checkpoint=args.checkpoint).eval()
    correct, seen = 0, 0
    for images, labels in tqdm(loader, desc=args.split):
        images, labels = images.to(device), labels.to(device)
        predictions = model.predict(images).argmax(dim=1)
        correct += int((predictions == labels).sum())
        seen += labels.shape[0]
    print(f"split={args.split} samples={seen} accuracy={correct / seen:.6f}")


if __name__ == "__main__":
    main()
