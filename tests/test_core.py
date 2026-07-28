from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from otf_cbm.backbones import foreground_patch_mask
from otf_cbm.checkpoints import (
    load_stage1_cost,
    load_stage2_checkpoint,
    save_stage2_checkpoint,
)
from otf_cbm.concepts import ConceptBank
from otf_cbm.datasets import CUBDataset
from otf_cbm.distributed import (
    DistributedContext,
    DistributedEvalSampler,
    resolve_seed,
)
from otf_cbm.models.cost import CostModel, compute_pairwise_phi
from otf_cbm.models.otf_cbm import OTFCBM
from otf_cbm.stage1 import resolve_paco_image
from otf_cbm.transport import unbalanced_sinkhorn_log
from scripts.preprocess_paco import mask_to_patch_ids
from scripts.train_iot import build_phases


class DummyBackbone(nn.Module):
    def __init__(self, dimension: int = 8) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.dimension = dimension

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        base = torch.arange(
            batch * 4 * self.dimension,
            device=images.device,
            dtype=images.dtype,
        ).reshape(batch, 4, self.dimension)
        patches = base / base.max().clamp_min(1.0)
        return {
            "x_norm_patchtokens": patches,
            "x_norm_clstoken": patches.mean(dim=1),
        }


def make_bank() -> ConceptBank:
    bank = ConceptBank(
        global_embeddings=torch.randn(7, 6),
        class_embeddings=[torch.randn(3, 6) for _ in range(3)],
        class_names=["zero", "one", "two"],
        global_texts=[f"concept {index}" for index in range(7)],
    )
    bank.validate()
    return bank


def make_model() -> OTFCBM:
    config = {
        "model": {
            "image_dim": 8,
            "feature_dim": 6,
            "adapter_hidden_dim": 4,
            "num_cost_bases": 18,
            "flow_hidden_dim": 12,
            "time_dim": 4,
            "clusters": 2,
            "cluster_iterations": 3,
            "cluster_backend": "torch",
            "foreground_keep_ratio": 0.5,
            "foreground_method": "cls_similarity",
            "background_ratio": 0.7,
            "background_penalty": 50.0,
            "class_concept_samples": 2,
            "activation_chunk_size": 3,
            "activation_top_regions": 2,
            "sinkhorn": {"epsilon": 0.1, "tau": 0.5, "iterations": 4},
        }
    }
    model = OTFCBM(DummyBackbone(), make_bank(), 3, config)
    model.freeze_stage1_ot()
    return model


def test_pairwise_phi_has_18_bases_and_is_finite() -> None:
    phi = compute_pairwise_phi(torch.randn(4, 6), torch.randn(3, 6))
    assert phi.shape == (4, 3, 18)
    assert torch.isfinite(phi).all()


def test_stage2_freezes_theta_but_not_adapter() -> None:
    model = make_model()
    assert not model.cost_model.theta.requires_grad
    assert all(
        parameter.requires_grad for parameter in model.cost_model.img_proj.parameters()
    )
    assert model.cost_model.img_adapter is not None
    assert all(
        parameter.requires_grad
        for parameter in model.cost_model.img_adapter.parameters()
    )
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    projected = model.cost_model.project(torch.randn(3, 8))
    conditioned = model.condition_adapter(torch.randn(3, 8))
    assert torch.allclose(projected.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(conditioned.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_stage1_partial_identity_shortcut_and_phases() -> None:
    cost = CostModel(
        image_dim=8,
        text_dim=6,
        adapter_hidden_dim=4,
        shortcut_init="partial_identity",
    )
    expected = torch.zeros(6, 8)
    expected[:6, :6] = torch.eye(6)
    assert torch.equal(cost.img_proj.shortcut.weight, expected)

    phases = build_phases(
        {
            "theta_only_epochs": 5,
            "adapter_only_epochs": 5,
            "joint_epochs": 10,
            "theta_learning_rate": 1e-4,
            "adapter_learning_rate": 1e-4,
            "joint_learning_rate": 5e-5,
        }
    )
    assert [phase["epochs"] for phase in phases] == [5, 5, 10]
    assert [phase["name"] for phase in phases] == [
        "theta_only",
        "adapter_only",
        "joint",
    ]


def test_sinkhorn_shape_and_finiteness() -> None:
    cost = torch.rand(2, 4, 3)
    plan = unbalanced_sinkhorn_log(
        cost,
        torch.ones(2, 4),
        torch.ones(2, 3),
        epsilon=0.1,
        tau=0.5,
        iterations=5,
    )
    assert plan.shape == cost.shape
    assert torch.isfinite(plan).all()


def test_dynamic_concept_dimension_and_separate_interfaces() -> None:
    model = make_model()
    images = torch.randn(2, 3, 4, 4)
    labels = torch.tensor([0, 2])
    logits, flow_loss, activations = model.forward_train(images, labels)
    assert logits.shape == (2, 3)
    assert activations.shape == (2, 7)
    assert flow_loss.ndim == 0
    inference_logits = model.predict(images)
    ddp_compatible_logits = model(images)
    ddp_train_logits, _, _ = model(images, labels)
    assert inference_logits.shape == (2, 3)
    assert ddp_compatible_logits.shape == (2, 3)
    assert ddp_train_logits.shape == (2, 3)
    assert model.classifier[0].normalized_shape == (7,)
    assert model.classifier[1].in_features == 7


def test_export_excludes_backbone() -> None:
    state = make_model().export_state_dict()
    assert state
    assert not any(key.startswith("backbone.") for key in state)


def test_legacy_stage1_key_conversion(tmp_path) -> None:
    source = CostModel(image_dim=8, text_dim=6, adapter_hidden_dim=4)
    legacy = {}
    for key, value in source.state_dict().items():
        key = key.replace("img_proj.norm.", "img_proj.ln.")
        key = key.replace("img_proj.shortcut.", "img_proj.short.")
        legacy[key] = value
    path = tmp_path / "legacy.pth"
    torch.save(legacy, path)
    target = CostModel(image_dim=8, text_dim=6, adapter_hidden_dim=4)
    load_stage1_cost(target, path)
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])


def test_compact_stage2_checkpoint_roundtrip(tmp_path) -> None:
    source = make_model()
    path = tmp_path / "stage2.pt"
    save_stage2_checkpoint(path, source, epoch=2, config={"name": "test"})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert not any(key.startswith("backbone.") for key in payload["model_state"])
    target = make_model()
    loaded = load_stage2_checkpoint(target, path)
    assert loaded["epoch"] == 2
    for key, value in source.export_state_dict().items():
        assert torch.equal(value, target.state_dict()[key].cpu())


def test_checkpoint_rejects_different_class_order(tmp_path) -> None:
    source = make_model()
    path = tmp_path / "stage2.pt"
    save_stage2_checkpoint(path, source, epoch=1, config={})
    target = make_model()
    target.concept_bank.class_names = ["one", "zero", "two"]
    with pytest.raises(ValueError, match="class ordering"):
        load_stage2_checkpoint(target, path)


def test_distributed_eval_sampler_has_exact_coverage() -> None:
    dataset = list(range(17))
    shards = [list(DistributedEvalSampler(dataset, rank, 3)) for rank in range(3)]
    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(17))
    assert len(flattened) == len(set(flattened))


def test_seed_resolution_supports_random_and_explicit_values() -> None:
    context = DistributedContext(False, 0, 0, 1, torch.device("cpu"))
    assert resolve_seed(1234, context) == 1234
    assert 0 <= resolve_seed("random", context) < 2**31


def test_cub_loader_applies_bounding_box(tmp_path) -> None:
    root = tmp_path / "CUB_200_2011"
    image_dir = root / "images" / "001.Class"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (20, 10), color="white").save(image_dir / "sample.jpg")
    (root / "images.txt").write_text("1 001.Class/sample.jpg\n", encoding="utf-8")
    (root / "image_class_labels.txt").write_text("1 1\n", encoding="utf-8")
    (root / "train_test_split.txt").write_text("1 1\n", encoding="utf-8")
    (root / "classes.txt").write_text("1 001.Class\n", encoding="utf-8")
    (root / "bounding_boxes.txt").write_text("1 5.0 2.0 10.0 6.0\n", encoding="utf-8")
    dataset = CUBDataset(
        root,
        "train",
        transform=lambda image: torch.tensor(image.size),
        crop_to_bbox=True,
    )
    size, label = dataset[0]
    assert size.tolist() == [10, 6]
    assert label == 0


def test_paco_mask_transform_matches_stretched_input() -> None:
    edge = np.zeros((100, 200), dtype=np.float32)
    edge[:, :10] = 1.0
    selected = mask_to_patch_ids(edge, 224, 14, 0.01)
    assert selected
    assert all(index % 16 == 0 for index in selected)

    tiny = np.zeros((1000, 1000), dtype=np.float32)
    tiny[500, 500] = 1.0
    assert len(mask_to_patch_ids(tiny, 224, 14, 0.01)) == 1


class _Attention(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.num_heads = heads
        self.scale = (dimension // heads) ** -0.5
        self.qkv = nn.Linear(dimension, 3 * dimension)


class _AttentionBlock(nn.Module):
    def __init__(self, dimension: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attn = _Attention(dimension, heads=2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens


class _AttentionBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_AttentionBlock()])

    def prepare_tokens_with_masks(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens


def test_attention_foreground_mask_without_custom_dino_fork() -> None:
    torch.manual_seed(2)
    tokens = torch.randn(2, 5, 4)
    mask = foreground_patch_mask(
        _AttentionBackbone(),
        tokens,
        patch_tokens=tokens[:, 1:],
        cls_token=tokens[:, 0],
        keep_ratio=0.5,
        method="attention",
    )
    assert mask.shape == (2, 4)
    assert mask.dtype == torch.bool


def test_paco_image_resolver_handles_coco_split_directories(tmp_path) -> None:
    image_root = tmp_path / "coco"
    image_path = image_root / "train2017" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    assert resolve_paco_image(image_root, "sample.jpg") == image_path
