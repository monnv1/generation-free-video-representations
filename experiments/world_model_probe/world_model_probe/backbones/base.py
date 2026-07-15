from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from PIL import Image


class VideoBackboneAdapter(ABC):
    """Frozen video backbone adapter returning DiT tokens [N, D]."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    @abstractmethod
    def extract_tokens(self, frames: list[Image.Image], prompt: str = "") -> torch.Tensor:
        """Return a CPU tensor with shape [N, D]."""

