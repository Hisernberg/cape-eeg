"""Soft-label loss and masked disagreement auxiliary loss (docs/01 section 7)."""
from __future__ import annotations

import torch

EPS = 1e-7


def soft_cross_entropy(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """-sum_k q_k log p_k on mixture probabilities (float32). Same optimum as KL(q||p)."""
    return -(q * torch.log(p.float().clamp_min(EPS))).sum(-1)


def kl_rows(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    pc = p.float().clamp_min(EPS); pc = pc / pc.sum(-1, keepdim=True)
    qs = q.float()
    t = torch.where(qs > 0, qs * (torch.log(qs.clamp_min(EPS)) - torch.log(pc)), torch.zeros_like(qs))
    return t.sum(-1)


def masked_disagreement_mse(d_hat: torch.Tensor | None, d: torch.Tensor) -> torch.Tensor:
    """Mean squared error over rows with an observed d (n>1). Differentiable zero when none exist."""
    if d_hat is None:
        return torch.zeros((), device=d.device)
    m = torch.isfinite(d)
    if not m.any():
        return (d_hat * 0.0).sum()
    return ((d_hat[m] - d[m]) ** 2).mean()


def total_loss(out: dict, batch: dict, aux_weight: float) -> tuple[torch.Tensor, dict]:
    ce = soft_cross_entropy(out["p"], batch["q"]).mean()
    aux = masked_disagreement_mse(out["d_hat"], batch["d"]) if aux_weight > 0 else torch.zeros((), device=ce.device)
    loss = ce + aux_weight * aux
    return loss, {"loss": float(loss.detach()), "soft_ce": float(ce.detach()), "aux_mse": float(aux.detach())}
