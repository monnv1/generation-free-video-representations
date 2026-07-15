from __future__ import annotations

import os
from typing import Any

import torch
from PIL import Image

from world_model_probe.backbones.base import VideoBackboneAdapter


class AttrDict(dict):
    """dict with attribute access, matching the subset starVLA wrappers expect."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({k: _to_attr_dict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(v) for v in value]
    return value


class _StarVLANativeBackbone(VideoBackboneAdapter):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.backbone_cfg = cfg["backbone"]
        self.device = torch.device(self.backbone_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.prompt = str(self.backbone_cfg.get("prompt", ""))
        self.model_id = self.backbone_cfg["model_id"]
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def _native_config(self) -> AttrDict:
        video_size = self.backbone_cfg.get("video_size", self.cfg["data"].get("image_size"))
        framework = {
            "world_model": {
                "base_wm": self.model_id,
                "extract_layers": self.backbone_cfg.get("extract_layers", [-1]),
            },
            "qwenvl": {
                "base_vlm": self.model_id,
            },
        }
        if video_size is not None:
            framework["obs_image_size"] = [int(video_size[0]), int(video_size[1])]
        return _to_attr_dict({"framework": framework})

    def _build_model(self) -> torch.nn.Module:
        if bool(self.backbone_cfg.get("local_files_only", False)):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
        from starVLA.model.modules.world_model import get_world_model

        return get_world_model(config=self._native_config())

    def extract_tokens(self, frames: list[Image.Image], prompt: str = "") -> torch.Tensor:
        prompt = prompt or self.prompt
        with torch.inference_mode():
            inputs = self.model.build_inputs([frames], [prompt])
            inputs = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in inputs.items()
            }
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            tokens = outputs.hidden_states[-1]
            if tokens.dim() != 3:
                raise ValueError(f"starVLA native wrapper returned tokens with shape {tuple(tokens.shape)}")
            return tokens[0].detach().to("cpu", dtype=torch.float32).contiguous()


class StarVLACosmosPredict2BackboneAdapter(_StarVLANativeBackbone):
    pass


class StarVLAWan22BackboneAdapter(_StarVLANativeBackbone):
    pass
