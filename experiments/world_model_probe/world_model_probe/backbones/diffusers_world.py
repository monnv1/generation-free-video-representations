from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from world_model_probe.backbones.base import VideoBackboneAdapter
from world_model_probe.utils import torch_dtype


class _DiffusersWorldBackbone(VideoBackboneAdapter):
    model_kind = ""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.backbone_cfg = cfg["backbone"]
        self.device = torch.device(self.backbone_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch_dtype(self.backbone_cfg.get("dtype", "bfloat16"))
        self.model_id = self.backbone_cfg["model_id"]
        self.prompt = str(self.backbone_cfg.get("prompt", ""))
        self.extract_layers = [int(x) for x in self.backbone_cfg.get("extract_layers", [-1])]
        self.video_size = self.backbone_cfg.get("video_size", cfg["data"].get("image_size", [256, 256]))
        self.num_frames = int(cfg["data"].get("input_frames", 8))
        self._intermediate_features: list[torch.Tensor] = []
        self._hooks = []
        self._load_components()
        self._register_hooks()
        self._freeze()

    def _freeze(self) -> None:
        for module in (self.text_encoder, self.vae, self.transformer):
            module.requires_grad_(False)
            module.eval()

    def _load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.backbone_cfg.get("cache_dir") is not None:
            kwargs["cache_dir"] = self.backbone_cfg["cache_dir"]
        if self.backbone_cfg.get("local_files_only") is not None:
            kwargs["local_files_only"] = bool(self.backbone_cfg["local_files_only"])
        return kwargs

    def _capture_hook(self, module, inputs, output) -> None:
        if isinstance(output, tuple):
            output = output[0]
        self._intermediate_features.append(output)

    def _flatten_feature(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 5:
            b, c, t, h, w = feat.shape
            feat = feat.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        if feat.dim() != 3:
            raise ValueError(f"Expected [B,N,D] or [B,C,T,H,W] hidden state, got {tuple(feat.shape)}")
        return feat

    def extract_tokens(self, frames: list[Image.Image], prompt: str = "") -> torch.Tensor:
        prompt = prompt or self.prompt
        with torch.inference_mode():
            inputs = self._build_inputs([frames], [prompt])
            self._intermediate_features.clear()
            out = self._forward_transformer(inputs)
            features = self._intermediate_features
            if not features:
                sample = out.sample if hasattr(out, "sample") else out
                if isinstance(sample, tuple):
                    sample = sample[0]
                features = [sample]
            tokens = self._flatten_feature(features[-1])[0]
        return tokens.detach().to("cpu", dtype=torch.float32).contiguous()


class CosmosPredict2BackboneAdapter(_DiffusersWorldBackbone):
    """Cosmos-Predict2 Video2World adapter using diffusers components."""

    model_kind = "cosmos"

    def _load_components(self) -> None:
        from diffusers import AutoencoderKLWan, CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler
        from diffusers.video_processor import VideoProcessor
        from transformers import T5EncoderModel, T5TokenizerFast

        load_kwargs = self._load_kwargs()
        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_id, subfolder="tokenizer", **load_kwargs)
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.model_id, subfolder="text_encoder", torch_dtype=self.dtype, **load_kwargs
        ).to(self.device)
        self.transformer = CosmosTransformer3DModel.from_pretrained(
            self.model_id, subfolder="transformer", torch_dtype=self.dtype, **load_kwargs
        ).to(self.device)
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=self.dtype, **load_kwargs
        ).to(self.device)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler", **load_kwargs
        )
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _register_hooks(self) -> None:
        blocks = self.transformer.transformer_blocks
        n_blocks = len(blocks)
        for idx in self.extract_layers:
            actual = idx if idx >= 0 else n_blocks + idx
            if actual < 0 or actual >= n_blocks:
                raise ValueError(f"Cosmos extract layer {idx} is out of range for {n_blocks} blocks.")
            self._hooks.append(blocks[actual].register_forward_hook(self._capture_hook))

    def _encode_text(self, prompts: list[str], max_length: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        text = self.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        ).last_hidden_state
        return text.to(dtype=self.dtype), text_inputs.attention_mask

    def _encode_video(self, videos: list[list[Image.Image]]) -> tuple[torch.Tensor, list[int]]:
        width, height = int(self.video_size[0]), int(self.video_size[1])
        tensors = []
        counts = []
        for frames in videos:
            tensor = self.video_processor.preprocess_video(frames, height=height, width=width)
            tensor = tensor.to(device=self.device, dtype=self.dtype)
            tensors.append(tensor)
            counts.append(tensor.shape[2])
        target_frames = self.num_frames
        batch = []
        for i, tensor in enumerate(tensors):
            n = tensor.shape[2]
            if n > target_frames:
                tensor = tensor[:, :, :target_frames]
                counts[i] = target_frames
            elif n < target_frames:
                pad = tensor[:, :, -1:].repeat(1, 1, target_frames - n, 1, 1)
                tensor = torch.cat([tensor, pad], dim=2)
            batch.append(tensor.squeeze(0))
        video = torch.stack(batch, dim=0)
        latents = self.vae.encode(video).latent_dist.sample()
        if self.vae.config.latents_mean is not None:
            mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
            std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1)
            mean = mean.to(device=latents.device, dtype=latents.dtype)
            std = std.to(device=latents.device, dtype=latents.dtype)
            latents = (latents - mean) / std * self.scheduler.config.sigma_data
        return latents, counts

    def _build_inputs(self, videos: list[list[Image.Image]], prompts: list[str]) -> dict[str, torch.Tensor]:
        text, text_mask = self._encode_text(prompts)
        latents, counts = self._encode_video(videos)
        batch_size = latents.shape[0]
        _, _, t_lat, h_lat, w_lat = latents.shape
        timestep = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        condition_mask = latents.new_zeros(batch_size, 1, t_lat, h_lat, w_lat)
        for i, n_cond in enumerate(counts):
            n_cond_latent = (n_cond - 1) // self.vae_scale_factor_temporal + 1
            condition_mask[i, :, :n_cond_latent] = 1.0
        padding_mask = latents.new_zeros(1, 1, h_lat, w_lat)
        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text,
            "attention_mask": text_mask,
            "condition_mask": condition_mask,
            "padding_mask": padding_mask,
        }

    def _forward_transformer(self, inputs: dict[str, torch.Tensor]) -> Any:
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            return self.transformer(
                hidden_states=inputs["hidden_states"],
                timestep=inputs["timestep"],
                encoder_hidden_states=inputs["encoder_hidden_states"],
                condition_mask=inputs["condition_mask"],
                padding_mask=inputs["padding_mask"],
            )


class Wan22BackboneAdapter(_DiffusersWorldBackbone):
    """Wan2.2 TI2V adapter using diffusers components."""

    model_kind = "wan"

    def _load_components(self) -> None:
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
        from diffusers.video_processor import VideoProcessor
        from transformers import T5TokenizerFast, UMT5EncoderModel

        load_kwargs = self._load_kwargs()
        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_id, subfolder="tokenizer", **load_kwargs)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.model_id, subfolder="text_encoder", torch_dtype=self.dtype, **load_kwargs
        ).to(self.device)
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.model_id, subfolder="transformer", torch_dtype=self.dtype, **load_kwargs
        ).to(self.device)
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=self.dtype, **load_kwargs
        ).to(self.device)
        self.scheduler = UniPCMultistepScheduler.from_pretrained(self.model_id, subfolder="scheduler", **load_kwargs)
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _register_hooks(self) -> None:
        blocks = self.transformer.blocks
        n_blocks = len(blocks)
        for idx in self.extract_layers:
            actual = idx if idx >= 0 else n_blocks + idx
            if actual < 0 or actual >= n_blocks:
                raise ValueError(f"Wan extract layer {idx} is out of range for {n_blocks} blocks.")
            self._hooks.append(blocks[actual].register_forward_hook(self._capture_hook))

    def _encode_text(self, prompts: list[str], max_length: int = 512) -> torch.Tensor:
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        text = self.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        ).last_hidden_state
        return text.to(dtype=self.dtype)

    def _encode_video(self, videos: list[list[Image.Image]]) -> torch.Tensor:
        width, height = int(self.video_size[0]), int(self.video_size[1])
        tensors = []
        for frames in videos:
            tensor = self.video_processor.preprocess_video(frames, height=height, width=width)
            tensors.append(tensor.to(device=self.device, dtype=self.dtype))
        target_frames = self.num_frames
        batch = []
        for tensor in tensors:
            n = tensor.shape[2]
            if n > target_frames:
                tensor = tensor[:, :, :target_frames]
            elif n < target_frames:
                pad = tensor[:, :, -1:].repeat(1, 1, target_frames - n, 1, 1)
                tensor = torch.cat([tensor, pad], dim=2)
            batch.append(tensor.squeeze(0))
        video = torch.stack(batch, dim=0)
        latents = self.vae.encode(video).latent_dist.sample()
        mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents = (latents - mean.to(latents.device, latents.dtype)) * inv_std.to(latents.device, latents.dtype)
        return latents

    def _build_inputs(self, videos: list[list[Image.Image]], prompts: list[str]) -> dict[str, torch.Tensor]:
        text = self._encode_text(prompts)
        latents = self._encode_video(videos)
        batch_size = latents.shape[0]
        p_t, p_h, p_w = self.transformer.config.patch_size
        _, _, t_lat, h_lat, w_lat = latents.shape
        seq_len = (t_lat // p_t) * (h_lat // p_h) * (w_lat // p_w)
        max_seq_len = int(self.backbone_cfg.get("max_seq_len", 1024))
        if seq_len > max_seq_len:
            raise ValueError(
                f"Wan seq_len={seq_len} exceeds max_seq_len={max_seq_len}; "
                "reduce data.input_frames or backbone.video_size."
            )
        timestep = torch.zeros(batch_size, seq_len, device=self.device, dtype=torch.long)
        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text,
        }

    def _forward_transformer(self, inputs: dict[str, torch.Tensor]) -> Any:
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            return self.transformer(
                hidden_states=inputs["hidden_states"],
                timestep=inputs["timestep"],
                encoder_hidden_states=inputs["encoder_hidden_states"],
            )
