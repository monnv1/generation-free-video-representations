#!/usr/bin/env python3
"""Quantify spatial alignment between a saved VAE latent delta and RGB change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr, spearmanr


def load_rgb(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def topk_metrics(a: np.ndarray, b: np.ndarray, fraction: float) -> dict[str, float]:
    count = max(1, round(a.size * fraction))
    a_idx = np.argpartition(a, -count)[-count:]
    b_idx = np.argpartition(b, -count)[-count:]
    overlap = len(np.intersect1d(a_idx, b_idx, assume_unique=False))
    union = 2 * count - overlap
    return {
        "fraction": fraction,
        "cells_per_set": count,
        "precision_recall": overlap / count,
        "iou": overlap / union,
        "random_expected_precision_recall": count / a.size,
        "random_expected_iou": (count / a.size) / (2.0 - count / a.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--current-frame", type=Path, required=True)
    parser.add_argument("--future-frame", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--current-key", default="current_latent")
    parser.add_argument("--future-key", default="true_t_plus_1")
    args = parser.parse_args()

    payload = torch.load(args.latents, map_location="cpu", weights_only=False)
    current = payload[args.current_key].float()
    future = payload[args.future_key].float()
    if current.shape != future.shape or current.ndim != 4:
        raise ValueError(f"Expected matching [B,C,H,W] tensors, got {current.shape} and {future.shape}")

    latent_delta = torch.linalg.vector_norm(future - current, dim=1, keepdim=True)
    current_rgb = load_rgb(args.current_frame)
    future_rgb = load_rgb(args.future_frame)
    if current_rgb.shape != future_rgb.shape:
        raise ValueError(f"Frame shapes differ: {current_rgb.shape} and {future_rgb.shape}")

    pixel_delta = torch.linalg.vector_norm(future_rgb - current_rgb, dim=1, keepdim=True)
    pixel_delta_grid = F.interpolate(
        pixel_delta,
        size=latent_delta.shape[-2:],
        mode="area",
    )

    latent_flat = latent_delta.numpy().ravel()
    pixel_flat = pixel_delta_grid.numpy().ravel()
    spearman = spearmanr(latent_flat, pixel_flat)
    pearson = pearsonr(latent_flat, pixel_flat)

    result = {
        "scope": "single saved frame pair; representation diagnostic only",
        "latent_file": args.latents.name,
        "current_frame": args.current_frame.name,
        "future_frame": args.future_frame.name,
        "latent_keys": [args.current_key, args.future_key],
        "latent_grid": list(latent_delta.shape[-2:]),
        "pixel_shape": list(current_rgb.shape[-2:]),
        "num_spatial_cells": int(latent_flat.size),
        "spearman_rho": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_pvalue": float(pearson.pvalue),
        "topk": [topk_metrics(latent_flat, pixel_flat, k) for k in (0.01, 0.05, 0.10)],
        "interpretation_limit": (
            "Measures spatial association with decoded RGB change. It does not establish "
            "linearity, causality, semantic disentanglement, or downstream control benefit."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
