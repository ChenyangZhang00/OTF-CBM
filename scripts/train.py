#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from otf_cbm.checkpoints import save_stage2_checkpoint
from otf_cbm.config import load_config, project_path
from otf_cbm.datasets import build_dataset
from otf_cbm.distributed import (
    DistributedContext,
    DistributedEvalSampler,
    barrier,
    cleanup_distributed,
    reduce_sums,
    setup_distributed,
    unwrap_model,
)
from otf_cbm.factory import build_stage2_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OTF-CBM Stage 2")
    parser.add_argument("--config", required=True)
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


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    context: DistributedContext,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    classification_weight = float(
        config["training"].get("classification_loss_weight", 1.0)
    )
    flow_weight = float(config["training"].get("flow_loss_weight", 1.0))
    loss_sum = 0.0
    classification_sum = 0.0
    flow_sum = 0.0
    correct = 0
    seen = 0
    progress = tqdm(
        loader,
        desc="train",
        disable=not context.is_main_process,
    )
    for images, labels in progress:
        images = images.to(context.device, non_blocking=True)
        labels = labels.to(context.device, non_blocking=True)
        logits, flow_loss, _ = model(images, labels)
        classification_loss = criterion(logits, labels)
        loss = classification_weight * classification_loss + flow_weight * flow_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clip = float(config["training"].get("gradient_clip", 0.0) or 0.0)
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                gradient_clip,
            )
        optimizer.step()

        batch_size = labels.shape[0]
        seen += batch_size
        loss_sum += float(loss.detach()) * batch_size
        classification_sum += float(classification_loss.detach()) * batch_size
        flow_sum += float(flow_loss.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum())
        if context.is_main_process:
            progress.set_postfix(
                loss=f"{loss_sum / seen:.4f}",
                accuracy=f"{100.0 * correct / seen:.2f}",
            )

    loss_sum, classification_sum, flow_sum, correct, seen = reduce_sums(
        [loss_sum, classification_sum, flow_sum, correct, seen],
        context,
    )
    return {
        "loss": loss_sum / seen,
        "classification_loss": classification_sum / seen,
        "flow_loss": flow_sum / seen,
        "accuracy": correct / seen,
        "samples": int(seen),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    context: DistributedContext,
) -> dict[str, float]:
    model = unwrap_model(model)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    classification_sum = 0.0
    flow_sum = 0.0
    correct = 0
    seen = 0
    progress = tqdm(
        loader,
        desc="validation",
        disable=not context.is_main_process,
    )
    for images, labels in progress:
        images = images.to(context.device, non_blocking=True)
        labels = labels.to(context.device, non_blocking=True)
        logits, flow_loss, _ = model(images, labels)
        classification_sum += float(criterion(logits, labels))
        flow_sum += float(flow_loss) * labels.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum())
        seen += labels.shape[0]

    classification_sum, flow_sum, correct, seen = reduce_sums(
        [classification_sum, flow_sum, correct, seen],
        context,
    )
    return {
        "classification_loss": classification_sum / seen,
        "flow_loss": flow_sum / seen,
        "accuracy": correct / seen,
        "samples": int(seen),
    }


def prepare_output(
    config: dict,
    context: DistributedContext,
) -> tuple[Path, Path]:
    output_dir = project_path(config, config["output"]["directory"])
    assert output_dir is not None
    if context.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(context)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(
            f"{metrics_path} already exists. Set output.directory to a new run."
        )
    return output_dir, metrics_path


def train(config: dict, context: DistributedContext) -> None:
    seed = int(config.get("seed", 1111))
    seed_everything(seed + context.rank)
    output_dir, metrics_path = prepare_output(config, context)

    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")
    if context.enabled:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        val_sampler = DistributedEvalSampler(
            val_dataset,
            rank=context.rank,
            world_size=context.world_size,
        )
    else:
        train_sampler = None
        val_sampler = None

    generator = torch.Generator()
    generator.manual_seed(seed + context.rank)
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": context.device.type == "cuda",
        "persistent_workers": int(config["training"]["num_workers"]) > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        **loader_options,
    )

    model = build_stage2_model(config, context.device)
    if context.enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            # Hard OT assignments can leave matching-only parameters unused.
            find_unused_parameters=True,
        )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    if context.is_main_process:
        trainable = sum(parameter.numel() for parameter in parameters)
        batch_per_device = int(config["training"]["batch_size"])
        print(
            f"distributed={context.enabled} world_size={context.world_size} "
            f"batch_per_device={batch_per_device} "
            f"global_batch={batch_per_device * context.world_size}"
        )
        print(f"trainable_parameters={trainable:,}")
        print(
            "frozen: backbone, theta; "
            "trainable: cost adapters, condition adapter, velocity field, classifier"
        )

    best_accuracy = -1.0
    epochs = int(config["training"]["epochs"])
    checkpoint_interval = int(config["training"].get("checkpoint_interval", 5))
    if epochs <= 0:
        raise ValueError("training.epochs must be positive")
    if checkpoint_interval <= 0:
        raise ValueError("training.checkpoint_interval must be positive")
    metrics_file = (
        metrics_path.open("w", encoding="utf-8") if context.is_main_process else None
    )
    try:
        for epoch in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            start = time.time()
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                config,
                context,
            )
            val_metrics = evaluate(model, val_loader, context)
            best = val_metrics["accuracy"] > best_accuracy
            if best:
                best_accuracy = val_metrics["accuracy"]

            metrics = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "best": best,
                "elapsed_minutes": (time.time() - start) / 60.0,
                "world_size": context.world_size,
            }

            if context.is_main_process:
                base_model = unwrap_model(model)
                save_stage2_checkpoint(
                    output_dir / "last.pt",
                    base_model,
                    epoch,
                    config,
                    optimizer=None,
                    metrics=metrics,
                )
                if epoch % checkpoint_interval == 0 or epoch == epochs:
                    save_stage2_checkpoint(
                        output_dir / f"epoch_{epoch:03d}.pt",
                        base_model,
                        epoch,
                        config,
                        optimizer=None,
                        metrics=metrics,
                    )
                if best:
                    save_stage2_checkpoint(
                        output_dir / "best.pt",
                        base_model,
                        epoch,
                        config,
                        optimizer=None,
                        metrics=metrics,
                    )
                assert metrics_file is not None
                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()
                print(
                    f"epoch={epoch} "
                    f"train_acc={train_metrics['accuracy']:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f} "
                    f"best_val_acc={best_accuracy:.4f}"
                )
            barrier(context)
    finally:
        if metrics_file is not None:
            metrics_file.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    context = setup_distributed(str(config.get("device", "cuda")))
    try:
        train(config, context)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
