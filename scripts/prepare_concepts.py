#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from otf_cbm.config import load_config, project_path, require_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode global and class-specific concepts with OpenCLIP"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        values = [line.strip() for line in handle if line.strip()]
    if not values:
        raise ValueError(f"No entries found in {path}")
    return values


@torch.inference_mode()
def encode(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    batches = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encoding concepts"):
        batch = texts[start : start + batch_size]
        batches.append(model.encode_text(tokenizer(batch).to(device)).float().cpu())
    return torch.cat(batches)


def main() -> None:
    args = parse_args()
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError(
            "Concept preparation requires open_clip_torch; install requirements.txt"
        ) from error
    config = load_config(args.config, args.set)
    concept_config = config["concepts"]
    global_texts = read_lines(
        require_file(config, concept_config["global_texts"], "Global concept file")
    )
    class_json_path = require_file(
        config, concept_config["class_concepts"], "Class concept JSON"
    )
    with class_json_path.open("r", encoding="utf-8") as handle:
        class_mapping = json.load(handle)
    if not isinstance(class_mapping, dict):
        raise TypeError("Class concept JSON must map class names to text lists")

    names_value = concept_config.get("class_names")
    if names_value:
        class_names = read_lines(
            require_file(config, names_value, "Ordered class-name file")
        )
    else:
        class_names = list(class_mapping)
    expected_classes = int(config["dataset"]["num_classes"])
    if len(class_names) != expected_classes:
        raise ValueError(
            f"Found {len(class_names)} class names, expected {expected_classes}. "
            "The order must match dataset label indices."
        )

    descriptions = []
    class_lengths = []
    for name in class_names:
        if name not in class_mapping:
            raise KeyError(f"Class {name!r} is missing from {class_json_path}")
        values = class_mapping[name]
        if isinstance(values, str):
            values = [values]
        values = [str(value).strip() for value in values if str(value).strip()]
        if not values:
            raise ValueError(f"Class {name!r} has no concepts")
        descriptions.extend(values)
        class_lengths.append(len(values))

    requested_device = config.get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    encoder_config = config["text_encoder"]
    pretrained = encoder_config["pretrained"]
    candidate = project_path(config, pretrained)
    if candidate is not None and candidate.exists():
        pretrained = str(candidate)
    model_name = encoder_config.get("model", "ViT-L-14")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    batch_size = int(concept_config.get("encoding_batch_size", 256))
    global_embeddings = encode(model, tokenizer, global_texts, batch_size, device)
    global_dtype = str(concept_config.get("global_dtype", "float32"))
    if global_dtype == "float16":
        global_embeddings = global_embeddings.half()
    elif global_dtype != "float32":
        raise ValueError("concepts.global_dtype must be float16 or float32")
    class_flat = encode(model, tokenizer, descriptions, batch_size, device)
    class_embeddings = list(torch.split(class_flat, class_lengths))

    output_path = project_path(config, concept_config["bank"])
    assert output_path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "encoder": {"model": model_name, "pretrained": str(pretrained)},
            "global_texts": global_texts,
            "global_embeddings": global_embeddings,
            "class_names": class_names,
            "class_embeddings": class_embeddings,
        },
        output_path,
    )
    print(
        f"saved={output_path} global_concepts={len(global_texts)} "
        f"classes={len(class_names)} embedding_dim={global_embeddings.shape[1]}"
    )


if __name__ == "__main__":
    main()
