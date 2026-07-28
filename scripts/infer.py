#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from otf_cbm.config import load_config
from otf_cbm.datasets import build_transform_from_config
from otf_cbm.factory import build_stage2_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run label-free image inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-classes", type=int, default=5)
    parser.add_argument("--top-concepts", type=int, default=10)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    requested_device = config.get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    model = build_stage2_model(config, device, checkpoint=args.checkpoint).eval()
    image_path = Path(args.image).expanduser().resolve()
    transform = build_transform_from_config(config, training=False)
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    logits, activations = model.predict(tensor, return_activations=True)
    probabilities = logits.softmax(dim=1)[0]

    class_count = min(args.top_classes, model.num_classes)
    class_scores, class_indices = probabilities.topk(class_count)
    print("Predicted classes:")
    for score, index in zip(class_scores.tolist(), class_indices.tolist()):
        name = model.concept_bank.class_names[index]
        print(f"  {name}: {score:.6f}")

    if model.concept_bank.global_texts:
        concept_count = min(args.top_concepts, model.num_concepts)
        concept_scores, concept_indices = activations[0].topk(concept_count)
        print("Most active concepts:")
        for score, index in zip(concept_scores.tolist(), concept_indices.tolist()):
            text = model.concept_bank.global_texts[index]
            print(f"  {text}: {score:.6f}")


if __name__ == "__main__":
    main()
