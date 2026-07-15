from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from world_model_probe.utils import import_object


@dataclass(frozen=True)
class BackboneInput:
    """The only inputs that a frozen world-model backbone should consume."""

    frames: list[Image.Image]
    semantic: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeSample:
    sample_id: str
    backbone_input: BackboneInput
    targets: dict[str, np.ndarray]
    valid: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


DataAdapter = Callable[[dict[str, Any], str], Iterable[ProbeSample]]


def resolve_data_adapter(cfg: dict[str, Any]) -> DataAdapter:
    adapter_path = str(cfg.get("data", {}).get("adapter", "world_model_probe.data.dom:iter_probe_samples"))
    adapter = import_object(adapter_path)
    if not callable(adapter):
        raise TypeError(f"Data adapter is not callable: {adapter_path}")
    return adapter
