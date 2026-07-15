from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPTrunk(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, depth: int = 2, dropout: float = 0.05) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for _ in range(max(1, int(depth))):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CurrentStateProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.trunk = MLPTrunk(input_dim, hidden_dim, depth, dropout)
        dim = self.trunk.output_dim
        self.xyz = nn.Linear(dim, 3)
        self.uv = nn.Linear(dim, 2)
        self.depth = nn.Linear(dim, 1)
        self.contact = nn.Linear(dim, 1)
        self.success = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk(x)
        return {
            "xyz": self.xyz(z),
            "uv": self.uv(z),
            "depth": self.depth(z),
            "contact_logits": self.contact(z).squeeze(-1),
            "success_logits": self.success(z).squeeze(-1),
        }


class FutureDynamicsProbe(nn.Module):
    def __init__(self, input_dim: int, num_horizons: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.num_horizons = int(num_horizons)
        self.trunk = MLPTrunk(input_dim, hidden_dim, depth, dropout)
        dim = self.trunk.output_dim
        h = self.num_horizons
        self.xyz = nn.Linear(dim, h * 3)
        self.uv = nn.Linear(dim, h * 2)
        self.depth = nn.Linear(dim, h)
        self.velocity_xyz = nn.Linear(dim, h * 3)
        self.contact = nn.Linear(dim, h)
        self.success = nn.Linear(dim, h)
        self.time_to_contact = nn.Linear(dim, h + 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk(x)
        b = x.shape[0]
        h = self.num_horizons
        return {
            "xyz": self.xyz(z).view(b, h, 3),
            "uv": self.uv(z).view(b, h, 2),
            "depth": self.depth(z).view(b, h, 1),
            "velocity_xyz": self.velocity_xyz(z).view(b, h, 3),
            "contact_logits": self.contact(z),
            "success_logits": self.success(z),
            "time_to_contact_logits": self.time_to_contact(z),
        }


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred - target).pow(2)
    if mask is None:
        return err.mean()
    while mask.dim() < err.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.to(err.dtype)
    denom = mask.sum().clamp_min(1.0) * err.shape[-1]
    return (err * mask).sum() / denom


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred - target).abs()
    if mask is None:
        return err.mean()
    while mask.dim() < err.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.to(err.dtype)
    denom = mask.sum().clamp_min(1.0) * err.shape[-1]
    return (err * mask).sum() / denom


def binary_metrics(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss = F.binary_cross_entropy_with_logits(logits, target)
    pred = (torch.sigmoid(logits) >= 0.5).to(target.dtype)
    acc = (pred == target).float().mean()
    return loss, acc
