from __future__ import annotations

import torch


def unbalanced_sinkhorn_log(
    cost: torch.Tensor,
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
    epsilon: float = 0.1,
    tau: float = 1.0,
    iterations: int = 50,
) -> torch.Tensor:
    """Batched log-domain unbalanced Sinkhorn used by the current code."""

    if cost.ndim == 2:
        cost = cost.unsqueeze(0)
        source_mass = source_mass.unsqueeze(0)
        target_mass = target_mass.unsqueeze(0)
        squeeze = True
    elif cost.ndim == 3:
        squeeze = False
    else:
        raise ValueError(f"cost must have 2 or 3 dimensions, got {cost.shape}")

    batch, sources, targets = cost.shape
    if source_mass.shape != (batch, sources):
        raise ValueError("source_mass shape does not match cost")
    if target_mass.shape != (batch, targets):
        raise ValueError("target_mass shape does not match cost")

    f = torch.zeros_like(source_mass)
    g = torch.zeros_like(target_mass)
    relaxation = tau / (tau + epsilon)

    def log_sum_exp(value: torch.Tensor, dim: int) -> torch.Tensor:
        return epsilon * torch.logsumexp(value / epsilon, dim=dim)

    for _ in range(iterations):
        f = relaxation * (
            torch.log(source_mass + 1e-8) - log_sum_exp(-cost + g.unsqueeze(1), dim=2)
        )
        g = relaxation * (
            torch.log(target_mass + 1e-8) - log_sum_exp(-cost + f.unsqueeze(2), dim=1)
        )

    plan = torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - cost) / epsilon)
    return plan.squeeze(0) if squeeze else plan
