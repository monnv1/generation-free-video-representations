from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from world_model_probe.model import target_keys_from_config
from world_model_probe.utils import read_jsonl


class LatentProbeDataset(Dataset):
    def __init__(self, index_path: str | Path, cfg: dict[str, Any] | None = None) -> None:
        self.index_path = Path(index_path)
        self.cfg = cfg
        self.target_mode = str((cfg or {}).get("targets", {}).get("mode", "absolute")).lower()
        if self.target_mode not in {"absolute", "delta"}:
            raise ValueError(f"targets.mode must be 'absolute' or 'delta', got {self.target_mode!r}.")
        self.target_keys = target_keys_from_config(cfg or {})
        self.expected_horizons = None
        if cfg is not None:
            self.expected_horizons = len(cfg.get("targets", {}).get("horizons", []))
        self._parquet_cache: dict[Path, Any] = {}
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing latent index: {self.index_path}")
        self.rows = read_jsonl(self.index_path)
        if not self.rows:
            raise ValueError(f"Latent index is empty: {self.index_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        blob = torch.load(row["latent_path"], map_location="cpu")
        tokens = blob["tokens"].float()
        cached_targets = {k: torch.as_tensor(v).float() for k, v in blob["targets"].items()}
        missing = [key for key in self.target_keys if key not in cached_targets]
        if missing:
            raise KeyError(f"Cached latent {row.get('latent_path')} is missing target key(s): {missing}")
        absolute_targets = {key: cached_targets[key] for key in self.target_keys}
        self._validate_target_shapes(row, absolute_targets)
        current_state = self._state_for_metrics(row, blob)
        targets = self._transform_targets(row, blob, absolute_targets, current_state)
        valid = torch.as_tensor(blob.get("valid", row.get("valid", []))).float()
        if valid.numel() == 0:
            valid = torch.ones(next(iter(targets.values())).shape[0], dtype=torch.float32)
        if valid.numel() != next(iter(targets.values())).shape[0]:
            raise ValueError(
                f"Cached latent {row.get('latent_path')} has valid mask length {valid.numel()} "
                f"but target horizon count {next(iter(targets.values())).shape[0]}."
            )
        return {
            "tokens": tokens,
            "targets": targets,
            "absolute_targets": absolute_targets,
            "current_state": current_state,
            "valid": valid,
            "metadata": row,
        }

    def _validate_target_shapes(self, row: dict[str, Any], targets: dict[str, torch.Tensor]) -> None:
        for key, value in targets.items():
            if value.dim() != 2:
                raise ValueError(
                    f"Cached target {key!r} in {row.get('latent_path')} must have shape [H,D], "
                    f"got {tuple(value.shape)}."
                )
            if self.expected_horizons is not None and value.shape[0] != self.expected_horizons:
                raise ValueError(
                    f"Cached target {key!r} in {row.get('latent_path')} has {value.shape[0]} horizons, "
                    f"but config targets.horizons has {self.expected_horizons}. "
                    "Regenerate latents with the current config or use a matching backbone.name/cache root."
                )

    def _transform_targets(
        self,
        row: dict[str, Any],
        blob: dict[str, Any],
        targets: dict[str, torch.Tensor],
        current_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.target_mode == "absolute":
            return targets
        current = current_state or self._current_state(row, blob)
        return {
            key: value - current[key].view(1, -1).expand_as(value)
            for key, value in targets.items()
        }

    def _state_for_metrics(self, row: dict[str, Any], blob: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.target_mode == "delta":
            return self._current_state(row, blob)
        if self.cfg is None:
            return {}
        try:
            return self._current_state(row, blob)
        except (FileNotFoundError, KeyError, ValueError):
            return {}

    def _current_state(self, row: dict[str, Any], blob: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.cfg is None:
            raise ValueError("targets.mode='delta' requires LatentProbeDataset(cfg=...).")
        path = self._parquet_path(row, blob)
        if path not in self._parquet_cache:
            self._parquet_cache[path] = pq.read_table(
                path,
                columns=["observation.environment_state", "observation.state"],
            ).to_pandas()
        df = self._parquet_cache[path]
        frame_index = int(row["frame_index"])
        env = torch.from_numpy(np.asarray(df.iloc[frame_index]["observation.environment_state"], dtype=np.float32).copy())
        arm = torch.from_numpy(np.asarray(df.iloc[frame_index]["observation.state"], dtype=np.float32).copy())
        target_cfg = self.cfg["targets"]
        return {
            "obj_pos": env[[int(i) for i in target_cfg.get("obj_pos_indices", [0, 1, 2])]],
            "obj_vel": env[[int(i) for i in target_cfg.get("obj_vel_indices", [6, 7, 8])]],
            "arm_pos": arm[[int(i) for i in target_cfg.get("arm_pos_indices", [0, 1, 2])]],
        }

    def _parquet_path(self, row: dict[str, Any], blob: dict[str, Any]) -> Path:
        metadata = blob.get("metadata", {})
        if "parquet_path" in metadata:
            return Path(metadata["parquet_path"])
        if self.cfg is None:
            raise ValueError("Cannot infer parquet_path without cfg.")
        dom_root = Path(self.cfg["data"]["dom_root"])
        chunk_id = int(self.cfg["data"].get("chunk_id", 0))
        episode_index = int(row["episode_index"])
        return dom_root / "data" / f"chunk-{chunk_id:03d}" / f"episode_{episode_index:06d}.parquet"


def collate_latent_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_tokens = max(item["tokens"].shape[0] for item in batch)
    dim = batch[0]["tokens"].shape[-1]
    tokens = torch.zeros(len(batch), max_tokens, dim, dtype=torch.float32)
    key_padding_mask = torch.ones(len(batch), max_tokens, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["tokens"].shape[0]
        tokens[i, :n] = item["tokens"]
        key_padding_mask[i, :n] = False
    target_keys = batch[0]["targets"].keys()
    targets = {k: torch.stack([item["targets"][k] for item in batch], dim=0) for k in target_keys}
    absolute_targets = {k: torch.stack([item["absolute_targets"][k] for item in batch], dim=0) for k in target_keys}
    current_state = {}
    if batch[0]["current_state"]:
        current_state = {
            k: torch.stack([item["current_state"][k] for item in batch], dim=0)
            for k in batch[0]["current_state"].keys()
        }
    valid = torch.stack([item["valid"] for item in batch], dim=0)
    return {
        "tokens": tokens,
        "key_padding_mask": key_padding_mask,
        "targets": targets,
        "absolute_targets": absolute_targets,
        "current_state": current_state,
        "valid": valid,
        "metadata": [item["metadata"] for item in batch],
    }


def infer_token_dim(index_path: str | Path) -> int:
    first = read_jsonl(index_path)[0]
    blob = torch.load(first["latent_path"], map_location="cpu")
    return int(blob["tokens"].shape[-1])
