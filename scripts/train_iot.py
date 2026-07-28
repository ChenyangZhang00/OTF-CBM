#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import secrets
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from otf_cbm.backbones import extract_dino_features, load_dinov2
from otf_cbm.checkpoints import save_stage1_checkpoint
from otf_cbm.config import load_config, project_path, require_file
from otf_cbm.models.cost import CostModel
from otf_cbm.stage1 import PACOImageLevelDataset, inverse_ot_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the shared PACO iOT cost")
    parser.add_argument("--config", default="configs/stage1_paco.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value; may be repeated",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_seed(value: int | str | None) -> int:
    if value is None or str(value).lower() == "random":
        return secrets.randbelow(2**31)
    seed = int(value)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed


def collate_records(records: list[dict]) -> list[dict]:
    return records


def build_text_encoder(config: dict, device: torch.device):
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError(
            "Stage-1 training requires open_clip_torch; install requirements.txt"
        ) from error
    text_config = config["text_encoder"]
    pretrained = text_config["pretrained"]
    candidate = project_path(config, pretrained)
    if candidate is not None and candidate.exists():
        pretrained = str(candidate)
    model, _, _ = open_clip.create_model_and_transforms(
        text_config.get("model", "ViT-L-14"), pretrained=pretrained
    )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(text_config.get("model", "ViT-L-14"))
    return model, tokenizer


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], device: torch.device):
    return model.encode_text(tokenizer(texts).to(device)).float()


def run_epoch(
    loader: DataLoader,
    backbone,
    text_model,
    tokenizer,
    cost_model: CostModel,
    config: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    regularize: bool,
) -> float:
    training = optimizer is not None
    cost_model.train(training)
    losses = []
    sinkhorn = config["training"]["sinkhorn"]
    progress = tqdm(loader, desc="train" if training else "validation")
    for records in progress:
        images = torch.stack([record["image"] for record in records]).to(device)
        with torch.no_grad():
            _, patch_batch = extract_dino_features(backbone, images)
            all_texts = [text for record in records for text in record["texts"]]
            encoded = encode_texts(text_model, tokenizer, all_texts, device)

        lengths = [len(record["texts"]) for record in records]
        text_batches = list(torch.split(encoded, lengths))
        sample_losses = []
        with torch.set_grad_enabled(training):
            for patches, texts, record in zip(patch_batch, text_batches, records):
                loss = inverse_ot_loss(
                    cost_model,
                    patches,
                    texts,
                    record["patch_ids"],
                    epsilon=float(sinkhorn["epsilon"]),
                    tau=float(sinkhorn["tau"]),
                    iterations=int(sinkhorn["iterations"]),
                    l1_weight=(
                        float(config["training"]["l1_weight"]) if regularize else 0.0
                    ),
                )
                if loss is not None:
                    if regularize:
                        adapter_weight_decay = float(
                            config["training"].get("adapter_weight_decay", 0.0)
                        )
                        if adapter_weight_decay:
                            loss = loss + adapter_weight_decay * sum(
                                parameter.square().sum()
                                for parameter in cost_model.img_proj.parameters()
                                if parameter.requires_grad
                            )
                    sample_losses.append(loss)
            if not sample_losses:
                continue
            batch_loss = torch.stack(sample_losses).mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    cost_model.parameters(),
                    float(config["training"].get("gradient_clip", 1.0)),
                )
                optimizer.step()
        losses.append(float(batch_loss.detach()))
        progress.set_postfix(loss=f"{np.mean(losses):.4f}")
    if not losses:
        raise RuntimeError("No valid PACO samples were found in this split")
    return float(np.mean(losses))


def set_trainable(
    cost_model: CostModel,
    train_theta: bool,
    train_adapter: bool,
) -> None:
    cost_model.theta.requires_grad_(train_theta)
    for parameter in cost_model.img_proj.parameters():
        parameter.requires_grad_(train_adapter)


def build_optimizer(
    cost_model: CostModel,
    train_theta: bool,
    train_adapter: bool,
    theta_learning_rate: float,
    adapter_learning_rate: float,
) -> torch.optim.Optimizer:
    groups = []
    if train_theta:
        groups.append({"params": [cost_model.theta], "lr": theta_learning_rate})
    if train_adapter:
        groups.append(
            {
                "params": list(cost_model.img_proj.parameters()),
                "lr": adapter_learning_rate,
            }
        )
    if not groups:
        raise ValueError("A Stage-1 phase must train theta or the adapter")
    return torch.optim.Adam(groups, betas=(0.9, 0.999))


def build_phases(training_config: dict) -> list[dict]:
    return [
        {
            "name": "theta_only",
            "epochs": int(training_config["theta_only_epochs"]),
            "train_theta": True,
            "train_adapter": False,
            "theta_learning_rate": float(training_config["theta_learning_rate"]),
            "adapter_learning_rate": float(training_config["adapter_learning_rate"]),
        },
        {
            "name": "adapter_only",
            "epochs": int(training_config["adapter_only_epochs"]),
            "train_theta": False,
            "train_adapter": True,
            "theta_learning_rate": float(training_config["theta_learning_rate"]),
            "adapter_learning_rate": float(training_config["adapter_learning_rate"]),
        },
        {
            "name": "joint",
            "epochs": int(training_config["joint_epochs"]),
            "train_theta": True,
            "train_adapter": True,
            "theta_learning_rate": float(training_config["joint_learning_rate"]),
            "adapter_learning_rate": float(training_config["joint_learning_rate"]),
        },
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    seed = resolve_seed(config.get("seed", "random"))
    config["seed"] = seed
    seed_everything(seed)
    print(f"seed={seed}")
    requested_device = config.get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    backbone = load_dinov2(config, device)
    text_model, tokenizer = build_text_encoder(config, device)

    data_config = config["data"]
    image_root = project_path(config, data_config["image_root"])
    assert image_root is not None
    train_dataset = PACOImageLevelDataset(
        require_file(config, data_config["train_json"], "PACO train JSON"),
        image_root,
        int(config.get("image_size", 224)),
        resize_mode=str(data_config.get("resize_mode", "stretch")),
    )
    val_dataset = PACOImageLevelDataset(
        require_file(config, data_config["val_json"], "PACO validation JSON"),
        image_root,
        int(config.get("image_size", 224)),
        resize_mode=str(data_config.get("resize_mode", "stretch")),
    )
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "collate_fn": collate_records,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model_config = config["model"]
    cost_model = CostModel(
        image_dim=int(model_config["image_dim"]),
        text_dim=int(model_config["feature_dim"]),
        adapter_hidden_dim=int(model_config["adapter_hidden_dim"]),
        num_bases=int(model_config.get("num_cost_bases", 18)),
        shortcut_init=str(model_config.get("shortcut_init", "partial_identity")),
    ).to(device)
    output_dir = project_path(config, config["output"]["directory"])
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(
            f"{metrics_path} already exists. Set output.directory to a new run."
        )

    training_config = config["training"]
    best_loss = float("inf")
    global_epoch = 0
    checkpoint_interval = int(training_config.get("checkpoint_interval", 1))
    if checkpoint_interval <= 0:
        raise ValueError("training.checkpoint_interval must be positive")
    phases = build_phases(training_config)
    if not any(phase["epochs"] > 0 for phase in phases):
        raise ValueError("At least one Stage-1 training phase must have epochs > 0")
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for phase in phases:
            if phase["epochs"] <= 0:
                continue
            set_trainable(
                cost_model,
                train_theta=phase["train_theta"],
                train_adapter=phase["train_adapter"],
            )
            optimizer = build_optimizer(
                cost_model,
                train_theta=phase["train_theta"],
                train_adapter=phase["train_adapter"],
                theta_learning_rate=phase["theta_learning_rate"],
                adapter_learning_rate=phase["adapter_learning_rate"],
            )
            for phase_epoch in range(1, phase["epochs"] + 1):
                global_epoch += 1
                start = time.time()
                train_loss = run_epoch(
                    train_loader,
                    backbone,
                    text_model,
                    tokenizer,
                    cost_model,
                    config,
                    device,
                    optimizer,
                    regularize=True,
                )
                val_loss = run_epoch(
                    val_loader,
                    backbone,
                    text_model,
                    tokenizer,
                    cost_model,
                    config,
                    device,
                    optimizer=None,
                    regularize=False,
                )
                metrics = {
                    "epoch": global_epoch,
                    "phase": phase["name"],
                    "phase_epoch": phase_epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "elapsed_minutes": (time.time() - start) / 60.0,
                    "seed": seed,
                }
                print(
                    f"epoch={global_epoch} phase={phase['name']} "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
                )
                save_stage1_checkpoint(
                    output_dir / "iot_last.pt",
                    cost_model,
                    global_epoch,
                    config,
                    optimizer=None,
                    metrics=metrics,
                )
                if (
                    global_epoch % checkpoint_interval == 0
                    or phase_epoch == phase["epochs"]
                ):
                    save_stage1_checkpoint(
                        output_dir / f"{phase['name']}_epoch_{global_epoch:03d}.pt",
                        cost_model,
                        global_epoch,
                        config,
                        optimizer=None,
                        metrics=metrics,
                    )
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_stage1_checkpoint(
                        output_dir / "iot_best.pt",
                        cost_model,
                        global_epoch,
                        config,
                        optimizer=None,
                        metrics=metrics,
                    )
                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()

    save_stage1_checkpoint(
        output_dir / "iot_final.pt",
        cost_model,
        global_epoch,
        config,
        optimizer=None,
        metrics={"best_val_loss": best_loss},
    )


if __name__ == "__main__":
    main()
