"""Future frozen-CLIP representation prediction head."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class FutureRepresentationHead(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 512) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, temporal: Tensor) -> Tensor:
        return F.normalize(self.projection(temporal), dim=-1)
