#!/usr/bin/env python3
"""Causal sigma probe for Cosmos-Predict2 Video2World.

This script runs two no-training diagnostics on one known video:

* A1: causal short-chain denoising from a pure-noise future slot.
* A0: sanity check where the true future latent is noised before one forward.

The implementation follows the Diffusers Cosmos2VideoToWorldPipeline
preconditioning code path, including c_in = c_skip = 1 / (sigma + 1),
c_out = -sigma / (sigma + 1), and clean condition-slot replacement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


def _add_optional_diffusers_cache_to_path() -> None:
    """Use the known local diffusers cache if the active env lacks diffusers."""
    try:
        import diffusers  # noqa: F401

        return
    except Exception:
        pass

    cache_root = Path("/data/uv-cache/archive-v0/MCt77ZsjPTHRGqy1")
    if cache_root.exists():
        sys.path.insert(0, str(cache_root))


_add_optional_diffusers_cache_to_path()

import cv2
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.video_processor import VideoProcessor
from PIL import Image
from transformers import T5EncoderModel, T5TokenizerFast


DEFAULT_MODEL_DIR = "/data/repos/starVLA/playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World"
DEFAULT_VIDEO = "/data/repos/cosmos-predict2.5/assets/base/robot_pouring.mp4"
DEFAULT_PROMPT = "/data/repos/cosmos-predict2.5/assets/base/robot_pouring.txt"
DEFAULT_OUTPUT_DIR = "/data/repos/cosmos_causal_probe/outputs/robot_pouring"


@dataclass
class ProbeConfig:
    model_dir: str
    video: str
    prompt: str
    negative_prompt: str
    output_dir: str
    frame_start: int
    frame_stride: int
    cond_frames: int
    future_pixel_frames: int
    height: int
    width: int
    fps: int
    sigmas: list[float]
    denoise_steps: list[int]
    guidance_scales: list[float]
    future_latent_slots: int
    seed: int
    dtype: str
    device: str
    hook_layer: int
    sigma_conditioning: float
    vae_sample_mode: str
    offload_encoders: bool
    save_latents: bool
    save_latent_dtype: str


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def read_prompt(path_or_text: str | Path) -> str:
    value = str(path_or_text)
    if value and Path(value).is_file():
        return Path(value).read_text(encoding="utf-8").strip()
    return value


def read_video_frames(path: str | Path, indices: Iterable[int]) -> list[Image.Image]:
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    frames: list[Image.Image] = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in indices:
        if idx < 0 or idx >= frame_count:
            raise IndexError(f"Requested frame {idx}, but video has {frame_count} frames")
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {idx} from {path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def cosine_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(a.shape[0], -1)
    b = b.detach().float().reshape(b.shape[0], -1)
    return float(F.cosine_similarity(a, b, dim=1).mean().item())


def mse_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.mse_loss(a.detach().float(), b.detach().float()).item())


def tag_float(value: float) -> str:
    text = f"{float(value):.8g}"
    return text.replace("-", "m").replace(".", "p")


def tensor_for_save(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=dtype).contiguous()


def save_tensor_bundle(path: Path, bundle: dict, dtype: torch.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, value in bundle.items():
        if isinstance(value, torch.Tensor):
            payload[key] = tensor_for_save(value, dtype)
        else:
            payload[key] = value
    torch.save(payload, path)


def save_reference_latents(
    output_dir: Path,
    cfg: ProbeConfig,
    latents: torch.Tensor,
    cond_latent_frames: int,
    target_latent_idx: int,
    frame_indices: list[int],
) -> Path:
    future_end = target_latent_idx + int(cfg.future_latent_slots)
    path = output_dir / "latents" / "reference_latents.pt"
    dtype = dtype_from_name(cfg.save_latent_dtype)
    save_tensor_bundle(
        path,
        {
            "all_latents": latents,
            "history_latents": latents[:, :, :cond_latent_frames],
            "current_latent": latents[:, :, cond_latent_frames - 1],
            "true_future_latents": latents[:, :, target_latent_idx:future_end],
            "cond_latent_frames": int(cond_latent_frames),
            "target_latent_idx": int(target_latent_idx),
            "target_latent_end": int(future_end),
            "future_latent_slots": int(cfg.future_latent_slots),
            "frame_indices": list(frame_indices),
            "condition_pixel_frame_indices": list(frame_indices[: cfg.cond_frames]),
            "future_pixel_frame_indices": list(frame_indices[cfg.cond_frames :]),
            "latent_shape": list(latents.shape),
            "saved_dtype": str(dtype).replace("torch.", ""),
        },
        dtype,
    )
    return path


def karras_schedule(sigma_start: float, sigma_end: float, steps: int, rho: float = 7.0) -> list[float]:
    if steps < 1:
        raise ValueError("denoise steps K must be >= 1")
    if sigma_start < sigma_end:
        raise ValueError(f"sigma_start={sigma_start} must be >= sigma_end={sigma_end}")
    ramp = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
    start = sigma_start ** (1.0 / rho)
    end = sigma_end ** (1.0 / rho)
    sigmas = (start + ramp * (end - start)) ** rho
    sigmas[0] = sigma_start
    sigmas[-1] = sigma_end
    return [float(x) for x in sigmas]


class CosmosCausalProbe:
    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = dtype_from_name(cfg.dtype)

        model_dir = cfg.model_dir
        self.tokenizer = T5TokenizerFast.from_pretrained(model_dir, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_dir, subfolder="text_encoder", torch_dtype=self.dtype
        ).eval()
        self.transformer = CosmosTransformer3DModel.from_pretrained(
            model_dir, subfolder="transformer", torch_dtype=self.dtype
        ).eval()
        self.vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=self.dtype).eval()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.transformer.to(self.device)
        self.text_encoder.to(self.device)
        self.vae.to(self.device)

        self._hook_features: list[torch.Tensor] = []
        self._hook_handle = None
        self._register_hook(cfg.hook_layer)

    def close(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _register_hook(self, layer_idx: int) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
        num_layers = len(self.transformer.transformer_blocks)
        actual = layer_idx if layer_idx >= 0 else num_layers + layer_idx
        if actual < 0 or actual >= num_layers:
            raise ValueError(f"hook_layer={layer_idx} resolves to {actual}, but model has {num_layers} blocks")

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self._hook_features.append(output.detach())

        self._hook_handle = self.transformer.transformer_blocks[actual].register_forward_hook(hook)

    @torch.no_grad()
    def encode_prompt(self, prompt: str, max_sequence_length: int = 512) -> torch.Tensor:
        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
            return_length=True,
            return_offsets_mapping=False,
        )
        attention_mask = text_inputs.attention_mask.bool().to(self.device)
        prompt_embeds = self.text_encoder(
            text_inputs.input_ids.to(self.device),
            attention_mask=attention_mask,
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

        lengths = attention_mask.sum(dim=1).cpu()
        for i, length in enumerate(lengths):
            prompt_embeds[i, int(length) :] = 0
        return prompt_embeds

    @torch.no_grad()
    def encode_video_latents(self, frames: list[Image.Image]) -> torch.Tensor:
        video = self.video_processor.preprocess_video(frames, height=self.cfg.height, width=self.cfg.width)
        video = video.to(device=self.device, dtype=self.dtype)

        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        enc = self.vae.encode(video)
        if self.cfg.vae_sample_mode == "mode" and hasattr(enc.latent_dist, "mode"):
            latents = enc.latent_dist.mode()
        else:
            latents = enc.latent_dist.sample(generator=generator)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device=self.device, dtype=latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device=self.device, dtype=latents.dtype)
        )
        latents = (latents - latents_mean) / latents_std * self.scheduler.config.sigma_data

        if self.cfg.offload_encoders and self.device.type == "cuda":
            self.text_encoder.to("cpu")
            self.vae.to("cpu")
            torch.cuda.empty_cache()

        return latents.to(self.device)

    def make_masks(self, latents: torch.Tensor, cond_latent_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _channels, num_latent_frames, height, width = latents.shape
        cond_indicator = latents.new_zeros(batch, 1, num_latent_frames, 1, 1)
        cond_indicator[:, :, :cond_latent_frames] = 1.0

        condition_mask = latents.new_zeros(batch, 1, num_latent_frames, height, width)
        condition_mask[:, :, :cond_latent_frames] = 1.0
        return cond_indicator, condition_mask.to(self.dtype)

    @torch.no_grad()
    def transformer_step(
        self,
        latents_full: torch.Tensor,
        conditioning_latents: torch.Tensor,
        cond_indicator: torch.Tensor,
        condition_mask: torch.Tensor,
        prompt_embeds: torch.Tensor,
        sigma: float,
        negative_prompt_embeds: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_tensor = torch.tensor(float(sigma), dtype=torch.float32, device=self.device)
        current_t = sigma_tensor / (sigma_tensor + 1.0)
        c_in = 1.0 - current_t
        c_skip = 1.0 - current_t
        c_out = -current_t

        cond_latent = latents_full * c_in
        cond_latent = cond_indicator * conditioning_latents + (1.0 - cond_indicator) * cond_latent
        cond_latent = cond_latent.to(self.dtype)

        timestep = current_t.view(1, 1, 1, 1, 1).expand(
            latents_full.size(0), -1, latents_full.size(2), -1, -1
        )
        sigma_conditioning = torch.tensor(self.cfg.sigma_conditioning, dtype=torch.float32, device=self.device)
        t_conditioning = sigma_conditioning / (sigma_conditioning + 1.0)
        cond_timestep = cond_indicator * t_conditioning + (1.0 - cond_indicator) * timestep
        cond_timestep = cond_timestep.to(self.dtype)

        padding_mask = latents_full.new_zeros(1, 1, self.cfg.height, self.cfg.width, dtype=self.dtype)

        self._hook_features.clear()
        raw_output = self.transformer(
            hidden_states=cond_latent,
            timestep=cond_timestep,
            encoder_hidden_states=prompt_embeds,
            fps=self.cfg.fps,
            condition_mask=condition_mask,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]
        if not self._hook_features:
            raise RuntimeError("No transformer hook features were captured")
        cond_hidden = self._hook_features[-1]

        pred_x0 = (c_skip * latents_full.float() + c_out * raw_output.float()).to(latents_full.dtype)
        pred_x0 = cond_indicator * conditioning_latents + (1.0 - cond_indicator) * pred_x0

        if guidance_scale > 1.0:
            if negative_prompt_embeds is None:
                raise ValueError("negative_prompt_embeds is required when guidance_scale > 1")
            raw_output_uncond = self.transformer(
                hidden_states=cond_latent,
                timestep=cond_timestep,
                encoder_hidden_states=negative_prompt_embeds,
                fps=self.cfg.fps,
                condition_mask=condition_mask,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]
            pred_x0_uncond = (c_skip * latents_full.float() + c_out * raw_output_uncond.float()).to(latents_full.dtype)
            pred_x0_uncond = cond_indicator * conditioning_latents + (1.0 - cond_indicator) * pred_x0_uncond
            pred_x0 = pred_x0 + float(guidance_scale) * (pred_x0 - pred_x0_uncond)

        return pred_x0, cond_hidden

    def pool_future_hidden(self, hidden: torch.Tensor, latents: torch.Tensor, target_latent_idx: int) -> torch.Tensor:
        _batch, _channels, _t_lat, h_lat, w_lat = latents.shape
        p_t, p_h, p_w = self.transformer.config.patch_size
        if p_t != 1:
            raise ValueError(f"This script expects temporal patch size 1, got {p_t}")
        tokens_per_latent_frame = (h_lat // p_h) * (w_lat // p_w)
        start = target_latent_idx * tokens_per_latent_frame
        end = start + tokens_per_latent_frame
        return hidden[:, start:end].float().mean(dim=1)

    def run_a1(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        cond_latent_frames: int,
        target_latent_idx: int,
        reference_hidden: torch.Tensor,
    ) -> list[dict]:
        cond_indicator, condition_mask = self.make_masks(latents, cond_latent_frames)
        conditioning_latents = latents.clone()
        future_slots = int(self.cfg.future_latent_slots)
        future_start = int(target_latent_idx)
        future_end = future_start + future_slots
        target_seq = latents[:, :, future_start:future_end].detach()
        target_first = latents[:, :, target_latent_idx].detach()
        prev_baseline = latents[:, :, cond_latent_frames - 1].detach()
        prev_seq = prev_baseline.unsqueeze(2).expand_as(target_seq)
        prev_cos = cosine_tensor(prev_baseline, target_first)
        prev_seq_cos = cosine_tensor(prev_seq, target_seq)

        rows: list[dict] = []
        sigma_min = float(getattr(self.scheduler.config, "sigma_min", 0.002))

        for guidance_scale in self.cfg.guidance_scales:
            for sigma_start in self.cfg.sigmas:
                for k in self.cfg.denoise_steps:
                    schedule = karras_schedule(float(sigma_start), sigma_min, int(k))
                    seed = self.cfg.seed + 1000 + int(sigma_start * 10000) + k + int(guidance_scale * 100) + future_slots * 17
                    gen = torch.Generator(device=self.device).manual_seed(seed)
                    x_future = torch.randn(target_seq.shape, device=self.device, dtype=latents.dtype, generator=gen)
                    x_future = x_future * float(sigma_start)
                    initial_x_future = x_future.detach().clone() if self.cfg.save_latents else None
                    initial_noise_cos = cosine_tensor(x_future[:, :, 0], target_first)
                    initial_noise_seq_cos = cosine_tensor(x_future, target_seq)

                    last_pred_seq = None
                    last_hidden = None
                    for sigma, sigma_next in zip(schedule[:-1], schedule[1:]):
                        latents_full = conditioning_latents.clone()
                        latents_full[:, :, future_start:future_end] = x_future
                        pred_x0, hidden = self.transformer_step(
                            latents_full=latents_full,
                            conditioning_latents=conditioning_latents,
                            cond_indicator=cond_indicator,
                            condition_mask=condition_mask,
                            prompt_embeds=prompt_embeds,
                            sigma=float(sigma),
                            negative_prompt_embeds=negative_prompt_embeds,
                            guidance_scale=float(guidance_scale),
                        )
                        pred_seq = pred_x0[:, :, future_start:future_end].detach()
                        model_output = (x_future.float() - pred_seq.float()) / max(float(sigma), 1e-12)
                        x_future = (x_future.float() + (float(sigma_next) - float(sigma)) * model_output).to(latents.dtype)
                        last_pred_seq = pred_seq
                        last_hidden = hidden

                    if last_pred_seq is None or last_hidden is None:
                        raise RuntimeError("A1 loop did not run")

                    last_pred_first = last_pred_seq[:, :, 0]
                    x_final_first = x_future[:, :, 0]
                    hidden_pool = self.pool_future_hidden(last_hidden, latents, target_latent_idx)
                    hidden_cos = cosine_tensor(hidden_pool, reference_hidden)
                    pred_cos = cosine_tensor(last_pred_first, target_first)
                    x_final_cos = cosine_tensor(x_final_first, target_first)
                    pred_seq_cos = cosine_tensor(last_pred_seq, target_seq)
                    x_final_seq_cos = cosine_tensor(x_future, target_seq)

                    row = {
                        "mode": "A1_causal",
                        "guidance_scale": float(guidance_scale),
                        "negative_prompt": self.cfg.negative_prompt,
                        "future_latent_slots": future_slots,
                        "sigma_start": float(sigma_start),
                        "denoise_steps": int(k),
                        "sigma_schedule": " ".join(f"{s:.8g}" for s in schedule),
                        "target_latent_idx": int(target_latent_idx),
                        "target_latent_end": int(future_end),
                        "prev_latent_baseline_cos": prev_cos,
                        "prev_sequence_baseline_cos": prev_seq_cos,
                        "initial_noise_cos": initial_noise_cos,
                        "initial_noise_sequence_cos": initial_noise_seq_cos,
                        "latent_cos_pred_x0": pred_cos,
                        "latent_mse_pred_x0": mse_tensor(last_pred_first, target_first),
                        "latent_cos_x_final": x_final_cos,
                        "latent_mse_x_final": mse_tensor(x_final_first, target_first),
                        "sequence_cos_pred_x0": pred_seq_cos,
                        "sequence_mse_pred_x0": mse_tensor(last_pred_seq, target_seq),
                        "sequence_cos_x_final": x_final_seq_cos,
                        "sequence_mse_x_final": mse_tensor(x_future, target_seq),
                        "cos_x_final_vs_baseline_diff": x_final_cos - prev_cos,
                        "cos_pred_x0_vs_baseline_diff": pred_cos - prev_cos,
                        "sequence_cos_x_final_vs_baseline_diff": x_final_seq_cos - prev_seq_cos,
                        "sequence_cos_pred_x0_vs_baseline_diff": pred_seq_cos - prev_seq_cos,
                        "hidden_cos": hidden_cos,
                    }

                    if self.cfg.save_latents:
                        latent_file = (
                            Path(self.cfg.output_dir)
                            / "latents"
                            / "a1"
                            / (
                                f"cfg_{tag_float(guidance_scale)}_"
                                f"sigma_{tag_float(sigma_start)}_"
                                f"K_{int(k)}_slots_{future_slots}.pt"
                            )
                        )
                        save_tensor_bundle(
                            latent_file,
                            {
                                "x_final_future_latents": x_future,
                                "pred_x0_future_latents": last_pred_seq,
                                "initial_noise_future_latents": initial_x_future,
                                "true_future_latents": target_seq,
                                "current_latent": prev_baseline,
                                "history_latents": latents[:, :, :cond_latent_frames],
                                "guidance_scale": float(guidance_scale),
                                "sigma_start": float(sigma_start),
                                "denoise_steps": int(k),
                                "sigma_schedule": [float(x) for x in schedule],
                                "cond_latent_frames": int(cond_latent_frames),
                                "target_latent_idx": int(target_latent_idx),
                                "target_latent_end": int(future_end),
                                "future_latent_slots": int(future_slots),
                            },
                            dtype_from_name(self.cfg.save_latent_dtype),
                        )
                        row["latents_file"] = str(latent_file)
                        row["reference_latents_file"] = str(Path(self.cfg.output_dir) / "latents" / "reference_latents.pt")

                    rows.append(row)
        return rows

    def run_a0(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        cond_latent_frames: int,
        target_latent_idx: int,
        reference_hidden: torch.Tensor,
    ) -> list[dict]:
        cond_indicator, condition_mask = self.make_masks(latents, cond_latent_frames)
        conditioning_latents = latents.clone()
        future_slots = int(self.cfg.future_latent_slots)
        future_start = int(target_latent_idx)
        future_end = future_start + future_slots
        target_seq = latents[:, :, future_start:future_end].detach()
        target = latents[:, :, target_latent_idx].detach()
        prev_baseline = latents[:, :, cond_latent_frames - 1].detach()
        prev_seq = prev_baseline.unsqueeze(2).expand_as(target_seq)
        prev_cos = cosine_tensor(prev_baseline, target)
        prev_seq_cos = cosine_tensor(prev_seq, target_seq)

        rows: list[dict] = []
        for guidance_scale in self.cfg.guidance_scales:
            for sigma in self.cfg.sigmas:
                gen = torch.Generator(device=self.device).manual_seed(
                    self.cfg.seed + 2000 + int(sigma * 10000) + int(guidance_scale * 100) + future_slots * 17
                )
                eps = torch.randn(target_seq.shape, device=self.device, dtype=latents.dtype, generator=gen)
                noisy_seq = ((1.0 - float(sigma)) * target_seq.float() + float(sigma) * eps.float()).to(latents.dtype)

                latents_full = conditioning_latents.clone()
                latents_full[:, :, future_start:future_end] = noisy_seq
                pred_x0, hidden = self.transformer_step(
                    latents_full=latents_full,
                    conditioning_latents=conditioning_latents,
                    cond_indicator=cond_indicator,
                    condition_mask=condition_mask,
                    prompt_embeds=prompt_embeds,
                    sigma=float(sigma),
                    negative_prompt_embeds=negative_prompt_embeds,
                    guidance_scale=float(guidance_scale),
                )
                pred_seq = pred_x0[:, :, future_start:future_end].detach()
                pred_future = pred_seq[:, :, 0]
                noisy_target = noisy_seq[:, :, 0]
                hidden_pool = self.pool_future_hidden(hidden, latents, target_latent_idx)
                pred_cos = cosine_tensor(pred_future, target)

                rows.append(
                    {
                        "mode": "A0_noised_true_future",
                        "guidance_scale": float(guidance_scale),
                        "negative_prompt": self.cfg.negative_prompt,
                        "future_latent_slots": future_slots,
                        "sigma_start": float(sigma),
                        "denoise_steps": 0,
                        "target_latent_idx": int(target_latent_idx),
                        "target_latent_end": int(future_end),
                        "prev_latent_baseline_cos": prev_cos,
                        "prev_sequence_baseline_cos": prev_seq_cos,
                        "noisy_input_cos": cosine_tensor(noisy_target, target),
                        "noisy_sequence_input_cos": cosine_tensor(noisy_seq, target_seq),
                        "latent_cos_pred_x0": pred_cos,
                        "latent_mse_pred_x0": mse_tensor(pred_future, target),
                        "sequence_cos_pred_x0": cosine_tensor(pred_seq, target_seq),
                        "sequence_mse_pred_x0": mse_tensor(pred_seq, target_seq),
                        "cos_pred_x0_vs_baseline_diff": pred_cos - prev_cos,
                        "sequence_cos_pred_x0_vs_baseline_diff": cosine_tensor(pred_seq, target_seq) - prev_seq_cos,
                        "hidden_cos": cosine_tensor(hidden_pool, reference_hidden),
                    }
                )
        return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(output_dir: Path, a1_rows: list[dict], a0_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_skipped.txt").write_text(f"matplotlib import failed: {exc}\n", encoding="utf-8")
        return

    if a1_rows:
        sigmas = sorted({float(row["sigma_start"]) for row in a1_rows})
        ks = sorted({int(row["denoise_steps"]) for row in a1_rows})
        heat = np.full((len(ks), len(sigmas)), np.nan, dtype=np.float32)
        for row in a1_rows:
            i = ks.index(int(row["denoise_steps"]))
            j = sigmas.index(float(row["sigma_start"]))
            value = float(row["cos_x_final_vs_baseline_diff"])
            heat[i, j] = value if np.isnan(heat[i, j]) else max(float(heat[i, j]), value)

        fig, ax = plt.subplots(figsize=(9, 3.8))
        im = ax.imshow(heat, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
        ax.set_yticks(range(len(ks)), [f"K={k}" for k in ks])
        ax.set_xlabel("sigma_start")
        ax.set_ylabel("denoise steps")
        ax.set_title("A1 x_final cosine gain over previous-latent baseline")
        for i in range(len(ks)):
            for j in range(len(sigmas)):
                value = heat[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "a1_gain_heatmap.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for k in ks:
            xs = []
            ys = []
            for sigma in sigmas:
                match = [
                    row
                    for row in a1_rows
                    if int(row["denoise_steps"]) == k and float(row["sigma_start"]) == sigma
                ]
                if match:
                    best_match = max(match, key=lambda row: float(row["cos_x_final_vs_baseline_diff"]))
                    xs.append(sigma)
                    ys.append(float(best_match["latent_cos_x_final"]))
            ax.plot(xs, ys, marker="o", label=f"A1 K={k}")
        baseline = float(a1_rows[0]["prev_latent_baseline_cos"])
        ax.axhline(baseline, linestyle="--", color="black", linewidth=1, label="prev latent baseline")
        ax.set_xscale("log")
        ax.set_xlabel("sigma_start")
        ax.set_ylabel("cos(x_final, target)")
        ax.set_title("A1 causal short-chain denoising")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "a1_x_final_curves.png", dpi=180)
        plt.close(fig)

    if a0_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = [float(row["sigma_start"]) for row in a0_rows]
        ys = [float(row["latent_cos_pred_x0"]) for row in a0_rows]
        noisy = [float(row["noisy_input_cos"]) for row in a0_rows]
        ax.plot(xs, ys, marker="o", label="A0 pred_x0")
        ax.plot(xs, noisy, marker="x", label="noisy input")
        baseline = float(a0_rows[0]["prev_latent_baseline_cos"])
        ax.axhline(baseline, linestyle="--", color="black", linewidth=1, label="prev latent baseline")
        ax.set_xscale("log")
        ax.set_xlabel("sigma")
        ax.set_ylabel("cosine to target latent")
        ax.set_title("A0 noised true future sanity check")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "a0_sanity_curve.png", dpi=180)
        plt.close(fig)


def summarize(output_dir: Path, cfg: ProbeConfig, latents: torch.Tensor, cond_latent_frames: int) -> None:
    summary = {
        "config": asdict(cfg),
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "latent_shape": list(latents.shape),
        "cond_latent_frames": cond_latent_frames,
        "target_latent_idx": cond_latent_frames,
        "vae_scale_factor_temporal": int((cfg.cond_frames + cfg.future_pixel_frames - 1) // (latents.shape[2] - 1))
        if latents.shape[2] > 1
        else None,
        "notes": [
            "A1 initializes all requested future latent slots from pure noise scaled by sigma_start.",
            "future_latent_slots controls how many future slots are denoised jointly with temporal self-attention.",
            "guidance_scale > 1 runs an additional negative-prompt branch and applies Cosmos CFG.",
            "K=1 means one transformer forward plus one Euler update to sigma_min.",
            "Cosmos pipeline preconditioning uses c_in=c_skip=1/(sigma+1), not EDM 1/sqrt(sigma^2+1).",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> ProbeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--cond-frames", type=int, default=5)
    parser.add_argument("--future-pixel-frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--sigmas", type=parse_csv_floats, default=parse_csv_floats("0.002,0.1,0.2,0.3,0.5,0.8,1.0"))
    parser.add_argument("--denoise-steps", type=parse_csv_ints, default=parse_csv_ints("1,3,5"))
    parser.add_argument("--guidance-scales", type=parse_csv_floats, default=parse_csv_floats("1.0"))
    parser.add_argument("--future-latent-slots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hook-layer", type=int, default=-1)
    parser.add_argument("--sigma-conditioning", type=float, default=0.0001)
    parser.add_argument("--vae-sample-mode", default="sample", choices=["sample", "mode"])
    parser.add_argument("--no-offload-encoders", action="store_true")
    parser.add_argument("--save-latents", action="store_true", help="Save reference and A1 denoised latents under output_dir/latents")
    parser.add_argument(
        "--save-latent-dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="dtype used for saved latent .pt files",
    )
    args = parser.parse_args()

    return ProbeConfig(
        model_dir=args.model_dir,
        video=args.video,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_dir=args.output_dir,
        frame_start=args.frame_start,
        frame_stride=args.frame_stride,
        cond_frames=args.cond_frames,
        future_pixel_frames=args.future_pixel_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        sigmas=args.sigmas,
        denoise_steps=args.denoise_steps,
        guidance_scales=args.guidance_scales,
        future_latent_slots=args.future_latent_slots,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        hook_layer=args.hook_layer,
        sigma_conditioning=args.sigma_conditioning,
        vae_sample_mode=args.vae_sample_mode,
        offload_encoders=not args.no_offload_encoders,
        save_latents=args.save_latents,
        save_latent_dtype=args.save_latent_dtype,
    )


def main() -> None:
    cfg = parse_args()
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_pixel_frames = cfg.cond_frames + cfg.future_pixel_frames
    frame_indices = [cfg.frame_start + i * cfg.frame_stride for i in range(total_pixel_frames)]
    prompt = read_prompt(cfg.prompt)
    negative_prompt = read_prompt(cfg.negative_prompt)
    frames = read_video_frames(cfg.video, frame_indices)

    probe = CosmosCausalProbe(cfg)
    try:
        prompt_embeds = probe.encode_prompt(prompt)
        negative_prompt_embeds = probe.encode_prompt(negative_prompt)
        latents = probe.encode_video_latents(frames)

        cond_latent_frames = (cfg.cond_frames - 1) // probe.vae_scale_factor_temporal + 1
        target_latent_idx = cond_latent_frames
        available_future_slots = latents.shape[2] - cond_latent_frames
        if target_latent_idx >= latents.shape[2] or cfg.future_latent_slots > available_future_slots:
            raise ValueError(
                f"Need {cfg.future_latent_slots} future latent slots, but only {available_future_slots} are available "
                f"with cond_latent_frames={cond_latent_frames}, T_lat={latents.shape[2]}. Increase --future-pixel-frames."
            )

        if cfg.save_latents:
            ref_path = save_reference_latents(output_dir, cfg, latents, cond_latent_frames, target_latent_idx, frame_indices)
            print(f"Saved reference latents to {ref_path}")

        cond_indicator, condition_mask = probe.make_masks(latents, cond_latent_frames)
        conditioning_latents = latents.clone()
        sigma_min = float(getattr(probe.scheduler.config, "sigma_min", 0.002))
        clean_reference_full = conditioning_latents.clone()
        clean_reference_full[:, :, target_latent_idx] = latents[:, :, target_latent_idx]
        _pred_ref, hidden_ref = probe.transformer_step(
            latents_full=clean_reference_full,
            conditioning_latents=conditioning_latents,
            cond_indicator=cond_indicator,
            condition_mask=condition_mask,
            prompt_embeds=prompt_embeds,
            sigma=sigma_min,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=1.0,
        )
        reference_hidden = probe.pool_future_hidden(hidden_ref, latents, target_latent_idx)

        a1_rows = probe.run_a1(
            latents, prompt_embeds, negative_prompt_embeds, cond_latent_frames, target_latent_idx, reference_hidden
        )
        a0_rows = probe.run_a0(
            latents, prompt_embeds, negative_prompt_embeds, cond_latent_frames, target_latent_idx, reference_hidden
        )

        write_csv(output_dir / "a1_results.csv", a1_rows)
        write_csv(output_dir / "a0_results.csv", a0_rows)
        summarize(output_dir, cfg, latents, cond_latent_frames)
        maybe_plot(output_dir, a1_rows, a0_rows)

        print(f"Wrote results to {output_dir}")
        if a1_rows:
            best = max(a1_rows, key=lambda row: float(row["cos_x_final_vs_baseline_diff"]))
            print("Best A1 by x_final gain:", json.dumps(best, indent=2))
    finally:
        probe.close()


if __name__ == "__main__":
    main()
