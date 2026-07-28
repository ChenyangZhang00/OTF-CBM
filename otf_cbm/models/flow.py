from __future__ import annotations

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim
        self.projection = nn.Linear(dim, dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.reshape(-1, 1)
        frequencies = torch.arange(
            self.dim // 2, device=time.device, dtype=time.dtype
        ).reshape(1, -1)
        angles = time * frequencies * (2.0 * math.pi)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.projection(embedding)


class VelocityField(nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        condition_dim: int = 768,
        time_dim: int = 32,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.network = nn.Sequential(
            nn.Linear(feature_dim + condition_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        nn.init.normal_(self.network[-1].weight, std=1e-4)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        point: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        embedded_time = self.time_embedding(time)
        return self.network(torch.cat([point, condition, embedded_time], dim=-1))
