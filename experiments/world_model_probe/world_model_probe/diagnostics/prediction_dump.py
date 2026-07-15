from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from world_model_probe.latent_dataset import LatentProbeDataset, collate_latent_samples
from world_model_probe.model import DynamicsProbe
from world_model_probe.utils import ensure_dir


TARGET_KEYS = ("obj_pos", "obj_vel", "arm_pos")


def _fmt(path: str, cfg: dict[str, Any]) -> str:
    return path.format(backbone=cfg["backbone"]["name"], run_name=cfg["project"]["run_name"])


def index_path(cfg: dict[str, Any], split: str) -> Path:
    return Path(_fmt(cfg["paths"]["latent_root"], cfg)) / f"index_{split}.jsonl"


def _append_np(chunks: dict[str, list[np.ndarray]], name: str, tensor: torch.Tensor) -> None:
    chunks.setdefault(name, []).append(tensor.detach().cpu().float().numpy())


def dump_predictions(
    cfg: dict[str, Any],
    checkpoint: str | Path,
    split: str,
    output_dir: str | Path,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: str | None = None,
    limit: int | None = None,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Run a trained probe on cached latents and save per-sample arrays."""
    output_dir = ensure_dir(output_dir)
    npz_path = output_dir / f"predictions_{split}.npz"
    metadata_path = output_dir / f"metadata_{split}.jsonl"
    summary_path = output_dir / f"summary_{split}.json"
    if npz_path.exists() and metadata_path.exists() and summary_path.exists() and not force:
        return npz_path, metadata_path, summary_path

    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    model_cfg = state.get("config", cfg)
    input_dim = int(state.get("input_dim", model_cfg["probe"].get("backbone_dim", cfg["probe"].get("backbone_dim"))))
    dev = torch.device(device or cfg.get("evaluation", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = DynamicsProbe(model_cfg, input_dim=input_dim).to(dev)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    ds = LatentProbeDataset(index_path(cfg, split), cfg)
    if limit is not None:
        ds.rows = ds.rows[: int(limit)]
    loader = DataLoader(
        ds,
        batch_size=int(batch_size or cfg.get("evaluation", {}).get("batch_size", 64)),
        shuffle=False,
        num_workers=int(num_workers if num_workers is not None else cfg.get("evaluation", {}).get("num_workers", 0)),
        pin_memory=dev.type == "cuda",
        collate_fn=collate_latent_samples,
    )

    chunks: dict[str, list[np.ndarray]] = {}
    metadata_rows: list[dict[str, Any]] = []
    target_mode = str(cfg.get("targets", {}).get("mode", "absolute")).lower()
    horizons = np.asarray([int(h) for h in cfg["targets"]["horizons"]], dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(dev, non_blocking=True)
            mask = batch["key_padding_mask"].to(dev, non_blocking=True)
            preds = model(tokens, key_padding_mask=mask)
            valid = batch["valid"].to(dev, non_blocking=True)
            _append_np(chunks, "valid", valid)

            for key in TARGET_KEYS:
                pred = preds[key].to(dev)
                gt_delta = batch["targets"][key].to(dev, non_blocking=True)
                gt_abs = batch["absolute_targets"][key].to(dev, non_blocking=True)
                current = batch["current_state"][key].to(dev, non_blocking=True)
                if target_mode == "delta":
                    pred_delta = pred
                    pred_abs = current.unsqueeze(1) + pred
                else:
                    pred_abs = pred
                    pred_delta = pred_abs - current.unsqueeze(1)
                _append_np(chunks, f"pred_delta_{key}", pred_delta)
                _append_np(chunks, f"gt_delta_{key}", gt_delta)
                _append_np(chunks, f"pred_abs_{key}", pred_abs)
                _append_np(chunks, f"gt_abs_{key}", gt_abs)
                _append_np(chunks, f"current_{key}", current)

            start = len(metadata_rows)
            for i, row in enumerate(batch["metadata"]):
                out = dict(row)
                out["array_index"] = start + i
                metadata_rows.append(out)

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    arrays["horizons"] = horizons
    np.savez_compressed(npz_path, **arrays)

    with open(metadata_path, "w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "checkpoint": str(checkpoint),
        "split": split,
        "num_samples": len(metadata_rows),
        "target_mode": target_mode,
        "horizons": horizons.tolist(),
        "arrays": {k: list(v.shape) for k, v in arrays.items()},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return npz_path, metadata_path, summary_path


def load_prediction_dump(npz_path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}
