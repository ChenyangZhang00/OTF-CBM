from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VisualAdapter(nn.Module):
    """The small DINOv2-to-CLIP adapter used by the current implementation."""

    def __init__(
        self,
        in_dim: int = 1024,
        out_dim: int = 768,
        hidden_dim: int = 512,
        use_residual: bool = True,
        normalize: bool = False,
        shortcut_init: str = "zero",
    ) -> None:
        super().__init__()
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.use_residual = use_residual
        self.normalize = normalize
        if use_residual:
            self.shortcut = nn.Linear(in_dim, out_dim, bias=False)
            if shortcut_init == "zero":
                nn.init.zeros_(self.shortcut.weight)
            elif shortcut_init == "partial_identity":
                nn.init.zeros_(self.shortcut.weight)
                size = min(in_dim, out_dim)
                with torch.no_grad():
                    self.shortcut.weight[:size, :size].copy_(torch.eye(size))
            else:
                raise ValueError("shortcut_init must be 'zero' or 'partial_identity'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.norm(self.proj2(self.act(self.proj1(x))))
        if self.use_residual:
            output = output + self.shortcut(x)
        if self.normalize:
            output = F.normalize(output, dim=-1)
        return output


class GeometryPreservingAdapter(nn.Module):
    """Small residual adapter applied only when computing the Stage-2 OT cost."""

    def __init__(self, dimension: int = 768, alpha_init: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(dimension, dimension, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        adapted = features + self.alpha * self.proj(features)
        return F.normalize(adapted, dim=-1)


def compute_pairwise_phi(
    visual: torch.Tensor,
    text: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute the 18 cost bases used in ``train_iot_paco.py``."""

    visual = visual.float()
    text = text.float()
    visual_norm2 = visual.square().sum(dim=1, keepdim=True)
    text_norm2 = text.square().sum(dim=1, keepdim=True)
    dot = visual @ text.T
    squared_distance = (visual_norm2 + text_norm2.T - 2.0 * dot).clamp_min(0.0)
    visual_norm = (visual_norm2 + eps).sqrt()
    text_norm = (text_norm2 + eps).sqrt()
    cosine = dot / (visual_norm @ text_norm.T + eps)
    distance = (squared_distance + eps).sqrt()

    bases = [
        torch.ones_like(dot),
        squared_distance,
        distance,
        torch.log1p(squared_distance),
        1.0 - cosine,
        (1.0 - cosine).square(),
        cosine,
        cosine.square(),
        dot,
    ]
    for sigma in (0.1, 0.5, 1.0):
        bases.append(torch.exp(-squared_distance / (2.0 * sigma**2)))
    bases.extend(
        [
            1.0 / (1.0 + squared_distance + eps),
            torch.exp(-distance),
            dot * torch.exp(-squared_distance / (2.0 * 0.5**2)),
            torch.exp(-distance / 0.5),
            (visual_norm - text_norm.T).abs(),
            (visual_norm - text_norm.T).abs() / (visual_norm + text_norm.T + eps),
        ]
    )

    phi = torch.stack(bases, dim=-1)
    flat = phi.reshape(-1, phi.shape[-1])
    mean = flat.mean(dim=0, keepdim=True)
    std = flat.std(dim=0, keepdim=True).clamp_min(1e-6)
    return ((flat - mean) / std).reshape_as(phi)


class CostModel(nn.Module):
    """Learned inverse-OT cost with a trainable visual adapter and 18 weights."""

    def __init__(
        self,
        image_dim: int = 1024,
        text_dim: int = 768,
        adapter_hidden_dim: int = 512,
        num_bases: int = 18,
        normalize_projection: bool = False,
        use_geometry_adapter: bool = False,
        geometry_alpha_init: float = 0.1,
        shortcut_init: str = "zero",
    ) -> None:
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.img_proj = VisualAdapter(
            in_dim=image_dim,
            out_dim=text_dim,
            hidden_dim=adapter_hidden_dim,
            use_residual=True,
            normalize=normalize_projection,
            shortcut_init=shortcut_init,
        )
        self.img_adapter = (
            GeometryPreservingAdapter(text_dim, alpha_init=geometry_alpha_init)
            if use_geometry_adapter
            else None
        )
        self.theta = nn.Parameter(torch.empty(num_bases))
        nn.init.normal_(self.theta, mean=0.0, std=0.01)

    def project(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.img_proj(image_features)

    def cost_from_projected(
        self,
        projected_visual: torch.Tensor,
        text_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.img_adapter is not None:
            projected_visual = self.img_adapter(projected_visual)
        visual = F.normalize(projected_visual, dim=-1)
        text = F.normalize(text_features, dim=-1)
        phi = compute_pairwise_phi(visual, text)
        cost = torch.einsum("nmb,b->nm", phi, self.theta)
        return cost, phi

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cost_from_projected(self.project(image_features), text_features)

    def freeze_ot_only(self) -> None:
        """Freeze only the learned OT cost weights; keep ``img_proj`` trainable."""

        self.theta.requires_grad_(False)
        for parameter in self.img_proj.parameters():
            parameter.requires_grad_(True)
        if self.img_adapter is not None:
            for parameter in self.img_adapter.parameters():
                parameter.requires_grad_(True)
