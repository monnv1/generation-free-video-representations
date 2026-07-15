from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from world_model_probe.backbones.base import VideoBackboneAdapter


class DryRunBackboneAdapter(VideoBackboneAdapter):
    """Cheap deterministic adapter for testing the cache/train/eval pipeline only."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        bcfg = cfg["backbone"]
        self.token_count = int(bcfg.get("token_count", 64))
        self.token_dim = int(bcfg.get("token_dim", cfg["probe"].get("backbone_dim", 2048)))
        seed = int(bcfg.get("seed", 17))
        gen = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(3, self.token_dim, generator=gen) / np.sqrt(3.0)

    def extract_tokens(self, frames: list[Image.Image], prompt: str = "") -> torch.Tensor:
        if not frames:
            raise ValueError("DryRunBackboneAdapter requires at least one frame.")
        arrs = []
        side = int(np.sqrt(self.token_count))
        side = max(1, side)
        for img in frames:
            small = img.resize((side, side), Image.BICUBIC)
            arr = np.asarray(small, dtype=np.float32) / 255.0
            arrs.append(arr.reshape(-1, 3))
        pixels = torch.from_numpy(np.concatenate(arrs, axis=0)).float()
        if pixels.shape[0] < self.token_count:
            repeat = self.token_count - pixels.shape[0]
            pixels = torch.cat([pixels, pixels[-1:].repeat(repeat, 1)], dim=0)
        pixels = pixels[: self.token_count]
        tokens = pixels @ self.proj
        return tokens.contiguous().cpu()

