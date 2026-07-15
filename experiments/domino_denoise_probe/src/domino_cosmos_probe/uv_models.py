from __future__ import annotations

import torch
import torch.nn as nn

from .probes import MLPTrunk


class UVProbe(nn.Module):
    """Small readout head that predicts only normalized UV positions."""

    def __init__(
        self,
        input_dim: int,
        num_points: int,
        hidden_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.trunk = MLPTrunk(input_dim, hidden_dim, depth, dropout)
        self.uv = nn.Linear(self.trunk.output_dim, self.num_points * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.uv(self.trunk(x)).view(x.shape[0], self.num_points, 2)
