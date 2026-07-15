from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from world_model_probe.model import probe_loss, target_keys_from_config


def _fmt(path: str, cfg: dict[str, Any]) -> str:
    return path.format(backbone=cfg["backbone"]["name"], run_name=cfg["project"]["run_name"])


def _index_path(cfg: dict[str, Any], split: str) -> Path:
    return Path(_fmt(cfg["paths"]["latent_root"], cfg)) / f"index_{split}.jsonl"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_payload(row: dict[str, Any]) -> dict[str, Any]:
    return torch.load(row["latent_path"], map_location="cpu")


def _target_means(cfg: dict[str, Any], train_index: Path, target_keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    count = 0
    parquet_cache: dict[Path, Any] = {}
    for row in _read_rows(train_index):
        payload = _load_payload(row)
        targets = _targets_for_mode(cfg, row, payload, parquet_cache, target_keys)
        for key in target_keys:
            value = targets[key].squeeze(0)
            sums[key] = value.clone() if key not in sums else sums[key] + value
        count += 1
    if count == 0:
        raise ValueError(f"Cannot compute train target means from empty index: {train_index}")
    return {key: value / count for key, value in sums.items()}


def _parquet_path(cfg: dict[str, Any], row: dict[str, Any], payload: dict[str, Any]) -> Path:
    metadata = payload.get("metadata", {})
    if "parquet_path" in metadata:
        return Path(metadata["parquet_path"])
    dom_root = Path(cfg["data"]["dom_root"])
    chunk_id = int(cfg["data"].get("chunk_id", 0))
    episode_index = int(row["episode_index"])
    return dom_root / "data" / f"chunk-{chunk_id:03d}" / f"episode_{episode_index:06d}.parquet"


def _current_state(
    cfg: dict[str, Any],
    row: dict[str, Any],
    payload: dict[str, Any],
    parquet_cache: dict[Path, Any],
) -> dict[str, torch.Tensor]:
    path = _parquet_path(cfg, row, payload)
    if path not in parquet_cache:
        parquet_cache[path] = pq.read_table(
            path,
            columns=["observation.environment_state", "observation.state"],
        ).to_pandas()
    df = parquet_cache[path]
    frame_index = int(row["frame_index"])
    env = torch.from_numpy(np.asarray(df.iloc[frame_index]["observation.environment_state"], dtype=np.float32).copy())
    arm = torch.from_numpy(np.asarray(df.iloc[frame_index]["observation.state"], dtype=np.float32).copy())
    target_cfg = cfg["targets"]
    return {
        "obj_pos": env[[int(i) for i in target_cfg.get("obj_pos_indices", [0, 1, 2])]],
        "obj_vel": env[[int(i) for i in target_cfg.get("obj_vel_indices", [6, 7, 8])]],
        "arm_pos": arm[[int(i) for i in target_cfg.get("arm_pos_indices", [0, 1, 2])]],
    }


def _expand_current(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return current.view(1, 1, -1).expand(1, target.shape[1], target.shape[2]).clone()


def _raw_targets(payload: dict[str, Any], target_keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(payload["targets"][key]).float().unsqueeze(0) for key in target_keys}


def _targets_for_mode(
    cfg: dict[str, Any],
    row: dict[str, Any],
    payload: dict[str, Any],
    parquet_cache: dict[Path, Any],
    target_keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    targets = _raw_targets(payload, target_keys)
    mode = str(cfg.get("targets", {}).get("mode", "absolute")).lower()
    if mode == "absolute":
        return targets
    if mode != "delta":
        raise ValueError(f"targets.mode must be 'absolute' or 'delta', got {mode!r}.")
    current = _current_state(cfg, row, payload, parquet_cache)
    return {key: targets[key] - _expand_current(current[key], targets[key]) for key in target_keys}


def _persistence_prediction(
    cfg: dict[str, Any],
    targets: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    target_keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    mode = str(cfg.get("targets", {}).get("mode", "absolute")).lower()
    if mode == "absolute":
        return {key: _expand_current(current[key], targets[key]) for key in target_keys}
    if mode == "delta":
        return {key: torch.zeros_like(targets[key]) for key in target_keys}
    raise ValueError(f"targets.mode must be 'absolute' or 'delta', got {mode!r}.")


def compute_baseline_metrics(cfg: dict[str, Any], split: str) -> dict[str, float]:
    """Compute zero, train-mean, and persistence baselines on a cached split."""
    target_keys = target_keys_from_config(cfg)
    split_index = _index_path(cfg, split)
    train_means = _target_means(cfg, _index_path(cfg, "train"), target_keys)
    weights = cfg["training"].get("loss_weights", {})
    loss_type = cfg["training"].get("loss_type", "smooth_l1")

    totals: dict[str, float] = {}
    count = 0
    parquet_cache: dict[Path, Any] = {}
    for row in _read_rows(split_index):
        payload = _load_payload(row)
        targets = _targets_for_mode(cfg, row, payload, parquet_cache, target_keys)
        valid = torch.as_tensor(payload.get("valid", row.get("valid", []))).float()
        if valid.numel() == 0:
            valid = torch.ones(next(iter(targets.values())).shape[1], dtype=torch.float32)
        valid = valid.unsqueeze(0)
        current = _current_state(cfg, row, payload, parquet_cache)
        preds_by_name = {
            "zero": {key: torch.zeros_like(targets[key]) for key in target_keys},
            "train_mean": {key: train_means[key].unsqueeze(0) for key in target_keys},
            "persistence": _persistence_prediction(cfg, targets, current, target_keys),
        }
        for name, preds in preds_by_name.items():
            _, metrics = probe_loss(preds, targets, valid, weights, loss_type)
            for key, value in metrics.items():
                metric_key = f"baseline/{name}/{key}"
                totals[metric_key] = totals.get(metric_key, 0.0) + float(value.item())
        count += 1

    if count == 0:
        raise ValueError(f"Cannot compute baseline metrics from empty index: {split_index}")
    metrics = {key: value / count for key, value in totals.items()}
    metrics["baseline/num_samples"] = count
    return metrics
