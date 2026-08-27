"""Reusable non-BC representation objectives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def future_info_nce(
    prediction: Tensor, target: Tensor, temperature: float = 0.07
) -> tuple[Tensor, dict[str, Tensor]]:
    """One-way in-batch InfoNCE from current prediction to frozen future target."""
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("prediction and target must both have shape [batch, feature]")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if prediction.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two in-batch examples")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    prediction = F.normalize(prediction, dim=-1)
    # Target is supervision only. Detach is deliberate and is the sole automatic
    # detach in R3; recurrent Mamba state is never implicitly detached.
    target = F.normalize(target.detach(), dim=-1)
    logits = prediction @ target.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        recall_at_1 = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, {"logits": logits, "recall_at_1": recall_at_1}
