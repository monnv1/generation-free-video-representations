from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image


DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)


def _torch_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


class CosmosDenoiseFeatureExtractor:
    """Frozen Cosmos-Predict2 feature extractor using the native Diffusers denoising loop."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        model_path = cfg["model_path"]
        self.device = torch.device(cfg.get("device", "cuda"))
        self.dtype = _torch_dtype(cfg.get("dtype", "bfloat16"))
        self.layers = [int(x) for x in cfg.get("layers", [8, 12, 16, 20, 24])]
        self.num_inference_steps = int(cfg.get("num_inference_steps", 35))
        self.capture_steps = [int(x) for x in cfg.get("capture_steps", [10, 14, 18, 22, 26])]
        self.cfg_scale = float(cfg.get("cfg_scale", 7.0))
        negative_prompt = cfg.get("negative_prompt", None)
        self.negative_prompt = DEFAULT_NEGATIVE_PROMPT if negative_prompt in {None, ""} else str(negative_prompt)
        self.fps = int(cfg.get("fps", 16))
        self.image_width = int(cfg.get("image_width", 320))
        self.image_height = int(cfg.get("image_height", 240))
        self.history_frames = int(cfg.get("history_frames", 5))
        self.future_frames = int(cfg.get("future_frames", 16))
        self.feature_pool = str(cfg.get("feature_pool", "future_steps")).lower()
        self.offload_text_vae = bool(cfg.get("offload_text_vae_after_encode", True))
        self.sigma_conditioning = float(cfg.get("sigma_conditioning", 0.0001))

        from diffusers import AutoencoderKLWan, CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler
        from diffusers.video_processor import VideoProcessor
        from transformers import T5EncoderModel, T5TokenizerFast

        self.tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
        )
        self.transformer = CosmosTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.vae = AutoencoderKLWan.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=self.dtype,
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.scheduler.register_to_config(
            sigma_max=80.0,
            sigma_min=0.002,
            sigma_data=1.0,
            final_sigmas_type="sigma_min",
        )
        self.video_processor = VideoProcessor(vae_scale_factor=8)

        self.text_encoder.requires_grad_(False).eval()
        self.transformer.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()

        self.hidden_size = int(self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim)
        self.patch_size = tuple(int(x) for x in getattr(self.transformer.config, "patch_size", [1, 2, 2]))
        self.temporal_factor = int(2 ** sum(bool(x) for x in self.vae.temperal_downsample))
        self._hooks: list[Any] = []
        self._captured: dict[int, torch.Tensor] = {}
        self._capture_enabled = False
        self._register_hooks()

    @property
    def source_names(self) -> list[str]:
        return ["raw_no_denoise"] + [f"denoise_step={step:03d}" for step in self.capture_steps]

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _register_hooks(self) -> None:
        num_blocks = len(self.transformer.transformer_blocks)

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                if not self._capture_enabled:
                    return
                feat = output[0] if isinstance(output, tuple) else output
                self._captured[layer_idx] = feat

            return hook

        for layer in self.layers:
            actual = layer if layer >= 0 else num_blocks + layer
            if not 0 <= actual < num_blocks:
                raise ValueError(f"Layer {layer} resolves to {actual}, outside 0..{num_blocks - 1}")
            self._hooks.append(self.transformer.transformer_blocks[actual].register_forward_hook(make_hook(layer)))

    def _move_text_vae_to_device(self) -> None:
        self.text_encoder.to(self.device)
        self.vae.to(self.device)

    def _offload_text_vae_to_cpu(self) -> None:
        if self.offload_text_vae:
            self.text_encoder.to("cpu")
            self.vae.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _encode_text(self, prompts: list[str]) -> torch.Tensor:
        self.text_encoder.to(self.device)
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        attention_mask = text_inputs.attention_mask.bool().to(self.device)
        with torch.inference_mode():
            embeds = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        embeds = embeds.to(dtype=self.dtype, device=self.device).clone()
        lengths = attention_mask.sum(dim=1).cpu()
        for i, length in enumerate(lengths):
            embeds[i, length:] = 0
        return embeds

    def _latent_frame_count(self, pixel_frames: int) -> int:
        return (int(pixel_frames) - 1) // self.temporal_factor + 1

    def _padding_mask(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.image_height, self.image_width, device=self.device, dtype=self.transformer.dtype)

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=self.dtype)
        return torch.autocast("cpu", enabled=False)

    def _prepare_timesteps(self) -> torch.Tensor:
        sigmas_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        sigmas = torch.linspace(0, 1, self.num_inference_steps, dtype=sigmas_dtype)
        self.scheduler.set_timesteps(sigmas=sigmas, device=self.device)
        if self.scheduler.config.final_sigmas_type == "sigma_min":
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]
        return self.scheduler.timesteps

    def _encode_conditioning_latents(self, frames: list[Image.Image]) -> tuple[torch.Tensor, int]:
        total_frames = self.history_frames + self.future_frames
        if len(frames) >= total_frames:
            video_frames = frames[-total_frames:]
            cond_latent_frames = self._latent_frame_count(total_frames)
        else:
            cond_latent_frames = self._latent_frame_count(len(frames))
            padding = [frames[-1]] * (total_frames - len(frames))
            video_frames = list(frames) + padding

        self.vae.to(self.device)
        video = self.video_processor.preprocess_video(
            video_frames,
            height=self.image_height,
            width=self.image_width,
        )
        video = video.to(device=self.device, dtype=self.vae.dtype)
        with torch.inference_mode():
            latents = self.vae.encode(video).latent_dist.sample()

        if self.vae.config.latents_mean is not None:
            mean = (
                torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=latents.dtype)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
            )
            std = (
                torch.tensor(self.vae.config.latents_std, device=self.device, dtype=latents.dtype)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
            )
            latents = (latents - mean) / std * float(self.scheduler.config.sigma_data)
        return latents.float(), cond_latent_frames

    def _condition_indicators(
        self,
        latent_shape: torch.Size,
        cond_latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _channels, t_lat, h_lat, w_lat = latent_shape
        indicator = torch.zeros(batch, 1, t_lat, 1, 1, device=self.device, dtype=torch.float32)
        indicator[:, :, :cond_latent_frames] = 1.0
        mask = indicator.expand(batch, 1, t_lat, h_lat, w_lat).to(self.transformer.dtype)
        return indicator, mask

    def _transformer_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        condition_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        capture: bool,
    ) -> torch.Tensor:
        self._captured.clear()
        self._capture_enabled = capture
        try:
            with torch.inference_mode(), self._autocast():
                out = self.transformer(
                    hidden_states=hidden_states.to(self.transformer.dtype),
                    timestep=timestep.to(self.transformer.dtype),
                    encoder_hidden_states=encoder_hidden_states.to(self.transformer.dtype),
                    fps=self.fps,
                    condition_mask=condition_mask.to(self.transformer.dtype),
                    padding_mask=padding_mask.to(self.transformer.dtype),
                    return_dict=False,
                )
        finally:
            self._capture_enabled = False
        return out[0]

    def _pool_feature(
        self,
        feat: torch.Tensor,
        *,
        latent_shape: torch.Size,
        cond_latent_frames: int,
        feature_pool: str,
    ) -> torch.Tensor:
        feat = feat.detach()
        if feat.dim() == 5:
            if feature_pool == "future_steps":
                feat = feat[:, :, cond_latent_frames:, :, :]
                return feat.float().mean(dim=(3, 4)).permute(0, 2, 1)
            if feature_pool == "future_only" and feat.shape[2] > cond_latent_frames:
                feat = feat[:, :, cond_latent_frames:, :, :]
            return feat.float().mean(dim=(2, 3, 4))

        if feat.dim() == 3:
            _b, _c, t_lat, h_lat, w_lat = latent_shape
            _pt, ph, pw = self.patch_size
            tokens_per_t = max(1, math.ceil(h_lat / ph) * math.ceil(w_lat / pw))
            if feature_pool == "future_steps":
                start = cond_latent_frames * tokens_per_t
                future_tokens = feat[:, start:, :]
                future_t = max(0, t_lat - cond_latent_frames)
                usable = min(future_tokens.shape[1], future_t * tokens_per_t)
                if usable <= 0:
                    raise ValueError("Captured feature has no future tokens to pool.")
                future_tokens = future_tokens[:, :usable, :]
                future_t = usable // tokens_per_t
                future_tokens = future_tokens[:, : future_t * tokens_per_t, :]
                return future_tokens.reshape(feat.shape[0], future_t, tokens_per_t, feat.shape[-1]).float().mean(dim=2)
            if feature_pool == "future_only":
                start = min(feat.shape[1], cond_latent_frames * tokens_per_t)
                if start < feat.shape[1]:
                    feat = feat[:, start:, :]
            return feat.float().mean(dim=1)

        raise ValueError(f"Unsupported captured feature shape: {tuple(feat.shape)}")

    def _pooled_captures(
        self,
        *,
        latent_shape: torch.Size,
        cond_latent_frames: int,
        feature_pool: str,
        repeat_steps: int | None = None,
    ) -> np.ndarray:
        pooled: list[np.ndarray] = []
        for layer in self.layers:
            if layer not in self._captured:
                raise RuntimeError(f"Layer {layer} was not captured.")
            tensor = self._pool_feature(
                self._captured[layer],
                latent_shape=latent_shape,
                cond_latent_frames=cond_latent_frames,
                feature_pool=feature_pool,
            )
            arr = tensor[0].cpu().numpy().astype(np.float32)
            if repeat_steps is not None and arr.ndim == 1:
                arr = np.repeat(arr[None, :], repeat_steps, axis=0)
            pooled.append(arr)
        return np.stack(pooled, axis=0)

    def extract_one(self, frames: list[Image.Image], prompt: str, seed: int) -> np.ndarray:
        if len(frames) != self.history_frames:
            raise ValueError(f"Expected {self.history_frames} frames, got {len(frames)}")
        invalid_steps = [step for step in self.capture_steps if step < 0 or step >= self.num_inference_steps]
        if invalid_steps:
            raise ValueError(f"capture_steps outside 0..{self.num_inference_steps - 1}: {invalid_steps}")

        self._move_text_vae_to_device()
        text = self._encode_text([prompt])
        negative_text = self._encode_text([self.negative_prompt])
        conditioning_latents, cond_latent_frames = self._encode_conditioning_latents(frames)
        self._offload_text_vae_to_cpu()

        total_latent_frames = conditioning_latents.shape[2]
        future_latent_frames = total_latent_frames - cond_latent_frames
        if future_latent_frames <= 0:
            raise ValueError("future_frames did not create any future latent slots.")

        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        latents = torch.randn(
            conditioning_latents.shape,
            device=self.device,
            dtype=torch.float32,
            generator=gen,
        )
        latents = latents * float(self.scheduler.config.sigma_max)

        indicator, cond_mask = self._condition_indicators(latents.shape, cond_latent_frames)
        uncond_indicator, uncond_mask = self._condition_indicators(latents.shape, cond_latent_frames)
        padding_mask = self._padding_mask()
        sigma_conditioning = torch.tensor(self.sigma_conditioning, dtype=torch.float32, device=self.device)
        t_conditioning = sigma_conditioning / (sigma_conditioning + 1)

        # Raw baseline: clean conditioning slots through the transformer at the native conditioning timestep.
        raw_timestep = torch.full(
            (latents.shape[0], 1, latents.shape[2], 1, 1),
            float(t_conditioning),
            device=self.device,
            dtype=torch.float32,
        )
        raw_hidden = indicator * conditioning_latents + (1 - indicator) * 0.0
        self._transformer_forward(
            hidden_states=raw_hidden,
            timestep=raw_timestep,
            encoder_hidden_states=text,
            condition_mask=cond_mask,
            padding_mask=padding_mask,
            capture=True,
        )
        raw = self._pooled_captures(
            latent_shape=latents.shape,
            cond_latent_frames=cond_latent_frames,
            feature_pool="all",
            repeat_steps=future_latent_frames if self.feature_pool == "future_steps" else None,
        )

        outputs_by_step: dict[int, np.ndarray] = {}
        timesteps = self._prepare_timesteps()
        self.scheduler._step_index = None
        capture_set = set(self.capture_steps)
        last_capture_step = max(capture_set)
        for i, t in enumerate(timesteps):
            current_sigma = self.scheduler.sigmas[i]
            current_t = current_sigma / (current_sigma + 1)
            c_in = 1 - current_t
            c_skip = 1 - current_t
            c_out = -current_t
            timestep = current_t.view(1, 1, 1, 1, 1).expand(latents.size(0), -1, latents.size(2), -1, -1)

            cond_latent = latents * c_in
            cond_latent = indicator * conditioning_latents + (1 - indicator) * cond_latent
            cond_timestep = indicator * t_conditioning + (1 - indicator) * timestep
            cond_capture = i in capture_set
            noise_pred = self._transformer_forward(
                hidden_states=cond_latent,
                timestep=cond_timestep,
                encoder_hidden_states=text,
                condition_mask=cond_mask,
                padding_mask=padding_mask,
                capture=cond_capture,
            )
            if cond_capture:
                outputs_by_step[i] = self._pooled_captures(
                    latent_shape=latents.shape,
                    cond_latent_frames=cond_latent_frames,
                    feature_pool=self.feature_pool,
                )

            noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(self.transformer.dtype)
            noise_pred = indicator * conditioning_latents + (1 - indicator) * noise_pred

            if self.cfg_scale > 1.0:
                uncond_latent = latents * c_in
                uncond_latent = uncond_indicator * conditioning_latents + (1 - uncond_indicator) * uncond_latent
                uncond_timestep = uncond_indicator * t_conditioning + (1 - uncond_indicator) * timestep
                noise_pred_uncond = self._transformer_forward(
                    hidden_states=uncond_latent,
                    timestep=uncond_timestep,
                    encoder_hidden_states=negative_text,
                    condition_mask=uncond_mask,
                    padding_mask=padding_mask,
                    capture=False,
                )
                noise_pred_uncond = (c_skip * latents + c_out * noise_pred_uncond.float()).to(self.transformer.dtype)
                noise_pred_uncond = uncond_indicator * conditioning_latents + (1 - uncond_indicator) * noise_pred_uncond
                noise_pred = noise_pred + self.cfg_scale * (noise_pred - noise_pred_uncond)

            if i >= last_capture_step:
                break
            model_output = (latents - noise_pred.float()) / current_sigma
            latents = self.scheduler.step(model_output, t, latents, return_dict=False)[0]

        missing = [step for step in self.capture_steps if step not in outputs_by_step]
        if missing:
            raise RuntimeError(f"Requested capture steps were not captured: {missing}")
        return np.stack([raw] + [outputs_by_step[step] for step in self.capture_steps], axis=0)
