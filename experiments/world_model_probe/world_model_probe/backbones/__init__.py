from __future__ import annotations

from world_model_probe.backbones.base import VideoBackboneAdapter
from world_model_probe.utils import import_object


def build_backbone(cfg: dict) -> VideoBackboneAdapter:
    backbone_cfg = cfg["backbone"]
    adapter_path = backbone_cfg["adapter"]
    cls = import_object(adapter_path)
    return cls(cfg)


__all__ = ["VideoBackboneAdapter", "build_backbone"]

