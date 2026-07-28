from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import project_path


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("DINOv2 checkpoint must be a state-dict-like mapping")
    for key in ("state_dict", "model", "teacher", "student"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    result = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        result[key] = value
    return result


def load_dinov2(config: dict[str, Any], device: torch.device) -> nn.Module:
    backbone_config = config["backbone"]
    source = backbone_config.get("source", "github")
    repository = backbone_config.get("repository", "facebookresearch/dinov2")
    if source == "local":
        resolved = project_path(config, repository)
        assert resolved is not None
        repository = str(resolved)

    weights_value = backbone_config.get("weights")
    weights_path = project_path(config, weights_value) if weights_value else None
    model = torch.hub.load(
        repository,
        backbone_config.get("name", "dinov2_vitl14"),
        source=source,
        pretrained=weights_path is None,
    )
    if weights_path is not None:
        if not weights_path.is_file():
            raise FileNotFoundError(f"DINOv2 weights not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(_unwrap_state_dict(checkpoint), strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "DINOv2 weights are incompatible with "
                f"{backbone_config.get('name', 'dinov2_vitl14')}: "
                f"{weights_path}"
            ) from error

    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def extract_dino_features(
    backbone: nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = backbone.forward_features(images)
    if not isinstance(output, dict):
        raise TypeError("DINOv2 forward_features must return a dictionary")
    try:
        patch_tokens = output["x_norm_patchtokens"]
        cls_token = output["x_norm_clstoken"]
    except KeyError as error:
        raise KeyError(
            "Expected x_norm_patchtokens and x_norm_clstoken from DINOv2"
        ) from error
    return cls_token, patch_tokens


def _manual_last_self_attention(
    backbone: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    prepare = getattr(backbone, "prepare_tokens_with_masks", None)
    blocks_value = getattr(backbone, "blocks", None)
    if not callable(prepare) or blocks_value is None:
        raise RuntimeError(
            "The loaded DINOv2 model does not expose the token and block APIs "
            "required for attention-based foreground masking."
        )

    blocks = list(blocks_value)
    if blocks and not hasattr(blocks[-1], "attn"):
        flattened = []
        for chunk in blocks:
            flattened.extend(block for block in chunk if hasattr(block, "attn"))
        blocks = flattened
    if not blocks:
        raise RuntimeError("The loaded DINOv2 model has no accessible blocks.")

    tokens = prepare(images)
    for block in blocks[:-1]:
        tokens = block(tokens)

    last_block = blocks[-1]
    normalized = last_block.norm1(tokens)
    attention_module = last_block.attn
    qkv = attention_module.qkv(normalized)
    batch, token_count, triple_dimension = qkv.shape
    heads = int(attention_module.num_heads)
    head_dimension = triple_dimension // (3 * heads)
    qkv = qkv.reshape(batch, token_count, 3, heads, head_dimension).permute(
        2, 0, 3, 1, 4
    )
    query, key = qkv[0], qkv[1]
    scale = float(getattr(attention_module, "scale", head_dimension**-0.5))
    return ((query * scale) @ key.transpose(-2, -1)).softmax(dim=-1)


@torch.no_grad()
def foreground_patch_mask(
    backbone: nn.Module,
    images: torch.Tensor,
    patch_tokens: torch.Tensor,
    cls_token: torch.Tensor,
    keep_ratio: float,
    method: str = "attention",
) -> torch.Tensor:
    """Return a per-patch foreground mask."""

    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")

    if method == "attention":
        attention = None
        for method_name in ("get_last_selfattention", "get_last_self_attention"):
            attention_method = getattr(backbone, method_name, None)
            if callable(attention_method):
                attention = attention_method(images)
                break
        if attention is None:
            attention = _manual_last_self_attention(backbone, images)
        patch_count = patch_tokens.shape[1]
        scores = attention[:, :, 0, -patch_count:].mean(dim=1)
    elif method == "cls_similarity":
        normalized_patches = torch.nn.functional.normalize(patch_tokens, dim=-1)
        normalized_cls = torch.nn.functional.normalize(cls_token, dim=-1)
        scores = torch.einsum("bnd,bd->bn", normalized_patches, normalized_cls)
    else:
        raise ValueError(
            f"Unsupported foreground method {method!r}; "
            "choose 'attention' or 'cls_similarity'."
        )

    threshold = torch.quantile(scores, q=1.0 - keep_ratio, dim=1, keepdim=True)
    return scores >= threshold
