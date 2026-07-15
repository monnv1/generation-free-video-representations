from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import dump_json, ensure_run_dir, load_config, load_json, set_seed
from .train_probes import _batch_indices, _safe_name, _standardize_features, _state_to_cpu
from .uv_models import UVProbe


def _as_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype)


def _uv_loss_and_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if target.dim() == 2:
        target = target.unsqueeze(1)
    if valid.dim() == 1:
        valid = valid.unsqueeze(1)
    valid = valid.to(pred.dtype)
    err = pred - target.to(pred.dtype)
    sq = err.pow(2).mean(dim=-1)
    abs_err = err.abs().mean(dim=-1)
    denom = valid.sum().clamp_min(1.0)
    loss = (sq * valid).sum() / denom
    mae_norm = (abs_err * valid).sum() / denom
    rmse_norm = torch.sqrt(loss.detach().clamp_min(0.0))

    scale = torch.tensor(
        [max(image_width - 1, 1), max(image_height - 1, 1)],
        device=pred.device,
        dtype=pred.dtype,
    )
    px_abs = (err.abs() * scale.view(1, 1, 2)).mean(dim=-1)
    mae_px = (px_abs * valid).sum() / denom
    metrics = {
        "loss": loss.detach(),
        "uv_mse_norm": loss.detach(),
        "uv_rmse_norm": rmse_norm,
        "uv_mae_norm": mae_norm.detach(),
        "uv_mae_px": mae_px.detach(),
        "valid_fraction": valid.mean().detach(),
    }
    return loss, metrics


def _evaluate(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    model.eval()
    accum: dict[str, float] = {}
    count = 0
    with torch.inference_mode():
        for idx in _batch_indices(indices, batch_size, shuffle=False, rng=np.random.default_rng(0)):
            pred = model(x[idx])
            _, metrics = _uv_loss_and_metrics(
                pred,
                target[idx],
                valid[idx],
                image_width=image_width,
                image_height=image_height,
            )
            b = len(idx)
            for key, value in metrics.items():
                accum[key] = accum.get(key, 0.0) + float(value.detach().cpu()) * b
            count += b
    return {key: value / max(1, count) for key, value in accum.items()}


def _train_one(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: dict,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float]], dict]:
    probe_cfg = cfg["uv_probe"]
    data_cfg = cfg["data"]
    batch_size = int(probe_cfg.get("batch_size", 128))
    epochs = int(probe_cfg.get("epochs", 60))
    patience = int(probe_cfg.get("patience", 10))
    image_width = int(data_cfg.get("image_width", 320))
    image_height = int(data_cfg.get("image_height", 240))
    save_optimizer = bool(probe_cfg.get("save_optimizer", False))
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(probe_cfg.get("lr", 3e-4)),
        weight_decay=float(probe_cfg.get("weight_decay", 1e-6)),
    )

    best_val = float("inf")
    best_state = None
    best_optimizer_state = None
    best_epoch = -1
    stale = 0
    curves: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for idx in _batch_indices(train_idx, batch_size, shuffle=True, rng=rng):
            pred = model(x[idx])
            loss, _metrics = _uv_loss_and_metrics(
                pred,
                target[idx],
                valid[idx],
                image_width=image_width,
                image_height=image_height,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += float(loss.detach().cpu()) * len(idx)
            train_count += len(idx)

        val_metrics = _evaluate(model, x, target, valid, val_idx, batch_size, image_width, image_height)
        val_loss = float(val_metrics["loss"])
        curve = {
            "epoch": float(epoch),
            "train_loss": train_loss_sum / max(1, train_count),
        }
        for key, value in val_metrics.items():
            curve[f"val_{key}"] = float(value)
        curves.append(curve)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_optimizer_state = _state_to_cpu(opt.state_dict()) if save_optimizer else None
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val = _evaluate(model, x, target, valid, val_idx, batch_size, image_width, image_height)
    test = _evaluate(model, x, target, valid, test_idx, batch_size, image_width, image_height)
    best_info = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "model_state_dict": best_state,
        "optimizer_state_dict": best_optimizer_state,
    }
    return val, test, curves, best_info


def _prepare_uv_targets(
    labels_npz,
    row_ids: np.ndarray,
    target_name: str,
    device: torch.device,
    horizon_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if target_name == "current_uv":
        target = labels_npz["current_uv"][row_ids]
        valid = labels_npz["current_uv_valid"][row_ids]
        num_points = 1
    elif target_name == "future_uv":
        target = labels_npz["future_uv"][row_ids]
        valid = labels_npz["future_uv_valid"][row_ids]
        if horizon_index is not None:
            target = target[:, horizon_index, :]
            valid = valid[:, horizon_index]
            num_points = 1
        else:
            num_points = target.shape[1]
    else:
        raise ValueError(f"Unsupported UV target: {target_name}")
    return _as_tensor(target, device), _as_tensor(valid, device), int(num_points)


def _future_step_for_horizon(horizon: int, temporal_factor: int, num_future_steps: int) -> int:
    # Slot k roughly covers the kth temporal-factor chunk after t. Use the last
    # available slot for horizons that land on the inclusive terminal frame.
    return min(max(math.ceil(int(horizon) / int(temporal_factor)) - 1, 0), num_future_steps - 1)


def train_uv_probe_grid(cfg: dict, run_dir: Path) -> tuple[Path, Path, Path]:
    meta = load_json(run_dir / "feature_meta.json")
    index = pd.read_parquet(run_dir / "slice_index.parquet").sort_values(["task", "episode_index", "t"]).reset_index(drop=True)
    row_ids = index["row_id"].to_numpy(dtype=np.int64)
    labels_npz = np.load(run_dir / "labels.npz")
    features = np.load(run_dir / "features.npy", mmap_mode="r")
    if tuple(features.shape) != tuple(meta["shape"]):
        raise ValueError(f"Feature shape mismatch: {features.shape} vs {meta['shape']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg.get("run", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    split = index["split"].to_numpy()
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(f"Need non-empty train/val/test splits; got {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    probe_cfg = cfg["uv_probe"]
    hidden_dim = int(probe_cfg.get("hidden_dim", 512))
    depth = int(probe_cfg.get("depth", 2))
    dropout = float(probe_cfg.get("dropout", 0.05))
    train_current_for_all = bool(probe_cfg.get("train_current_for_all_sources", True))
    ckpt_dir = run_dir / str(probe_cfg.get("checkpoint_dir", "uv_probe_ckpts"))
    save_checkpoints = bool(probe_cfg.get("save_checkpoints", True))
    save_current_checkpoints = bool(probe_cfg.get("save_current_checkpoints", True))
    if save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    horizons = labels_npz["horizons"].astype(int).tolist()
    per_step_features = features.ndim == 5
    temporal_factor = int(meta.get("temporal_factor", 4))
    num_future_steps = int(features.shape[3]) if per_step_features else 0

    records: list[dict] = []
    curve_records: list[dict] = []
    all_metrics: dict[str, dict] = {}
    for source_i, source in enumerate(tqdm(meta["sources"], desc="UV probe source")):
        for layer_i, layer in enumerate(meta["layers"]):
            jobs: list[dict] = []
            if per_step_features:
                for horizon_i, horizon in enumerate(horizons):
                    step_i = _future_step_for_horizon(horizon, temporal_factor, num_future_steps)
                    jobs.append(
                        {
                            "target_name": "future_uv",
                            "horizon_index": horizon_i,
                            "horizon": horizon,
                            "future_step": step_i,
                        }
                    )
                if train_current_for_all or source == "raw_no_denoise":
                    jobs.append(
                        {
                            "target_name": "current_uv",
                            "horizon_index": None,
                            "horizon": None,
                            "future_step": 0,
                        }
                    )
            else:
                jobs.append({"target_name": "future_uv", "horizon_index": None, "horizon": None, "future_step": None})
                if train_current_for_all or source == "raw_no_denoise":
                    jobs.append({"target_name": "current_uv", "horizon_index": None, "horizon": None, "future_step": None})

            for job in jobs:
                target_name = str(job["target_name"])
                future_step = job["future_step"]
                horizon_index = job["horizon_index"]
                horizon = job["horizon"]
                if per_step_features:
                    x_base = features[:, source_i, layer_i, int(future_step), :]
                else:
                    x_base = features[:, source_i, layer_i, :]
                x_np, mean, std = _standardize_features(x_base, train_idx)
                x = _as_tensor(x_np, device)
                input_dim = int(x.shape[-1])

                target, valid, num_points = _prepare_uv_targets(
                    labels_npz,
                    row_ids,
                    target_name,
                    device,
                    horizon_index=int(horizon_index) if horizon_index is not None else None,
                )
                model = UVProbe(input_dim, num_points, hidden_dim, depth, dropout)
                val, test, curves, best_info = _train_one(
                    model=model,
                    x=x,
                    target=target,
                    valid=valid,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    cfg=cfg,
                    device=device,
                    rng=rng,
                )
                suffix = ""
                if horizon is not None:
                    suffix = f"__horizon_{int(horizon)}__step_{int(future_step) + 1}"
                key = f"{target_name}|{source}|layer={layer}{suffix}"
                ckpt_path = None
                should_save_ckpt = save_checkpoints and (target_name != "current_uv" or save_current_checkpoints)
                if should_save_ckpt:
                    ckpt_path = ckpt_dir / f"{target_name}{suffix}__{_safe_name(source)}__layer_{layer}.pt"
                    payload = {
                        "target": target_name,
                        "source": source,
                        "source_index": source_i,
                        "layer": layer,
                        "layer_index": layer_i,
                        "future_step": int(future_step) if future_step is not None else None,
                        "future_step_1based": int(future_step) + 1 if future_step is not None else None,
                        "horizon": int(horizon) if horizon is not None else None,
                        "horizon_index": int(horizon_index) if horizon_index is not None else None,
                        "model_class": type(model).__name__,
                        "input_dim": input_dim,
                        "hidden_dim": hidden_dim,
                        "depth": depth,
                        "dropout": dropout,
                        "num_points": num_points,
                        "horizons": [int(horizon)] if horizon is not None else (horizons if target_name == "future_uv" else None),
                        "feature_mean": mean.astype(np.float32),
                        "feature_std": std.astype(np.float32),
                        "best_epoch": best_info["best_epoch"],
                        "best_val_loss": best_info["best_val_loss"],
                        "best_val_metrics": val,
                        "test_metrics": test,
                        "model_state_dict": best_info["model_state_dict"],
                    }
                    if best_info.get("optimizer_state_dict") is not None:
                        payload["optimizer_state_dict"] = best_info["optimizer_state_dict"]
                    torch.save(payload, ckpt_path)

                for curve in curves:
                    curve_record = {
                        "target": target_name,
                        "source": source,
                        "layer": layer,
                        "horizon": int(horizon) if horizon is not None else None,
                        "future_step": int(future_step) if future_step is not None else None,
                        "future_step_1based": int(future_step) + 1 if future_step is not None else None,
                    }
                    curve_record.update(curve)
                    curve_records.append(curve_record)

                all_metrics[key] = {
                    "target": target_name,
                    "source": source,
                    "layer": layer,
                    "horizon": int(horizon) if horizon is not None else None,
                    "horizon_index": int(horizon_index) if horizon_index is not None else None,
                    "future_step": int(future_step) if future_step is not None else None,
                    "future_step_1based": int(future_step) + 1 if future_step is not None else None,
                    "val": val,
                    "test": test,
                    "best_epoch": best_info["best_epoch"],
                    "best_val_loss": best_info["best_val_loss"],
                    "checkpoint": str(ckpt_path) if ckpt_path is not None else None,
                    "feature_mean_norm": float(np.linalg.norm(mean.astype(np.float32))),
                    "feature_std_mean": float(std.astype(np.float32).mean()),
                }
                for split_name, metrics in [("val", val), ("test", test)]:
                    rec = {
                        "target": target_name,
                        "source": source,
                        "layer": layer,
                        "horizon": int(horizon) if horizon is not None else None,
                        "future_step": int(future_step) if future_step is not None else None,
                        "future_step_1based": int(future_step) + 1 if future_step is not None else None,
                        "split": split_name,
                    }
                    rec.update(metrics)
                    records.append(rec)

    metrics_path = run_dir / "uv_probe_metrics.json"
    csv_path = run_dir / "uv_per_source_layer_metrics.csv"
    curves_path = run_dir / "uv_probe_training_curves.csv"
    dump_json(all_metrics, metrics_path)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    pd.DataFrame(curve_records).to_csv(curves_path, index=False)
    return metrics_path, csv_path, curves_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UV-only current/future probes over cached Cosmos features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("run", {}).get("seed", 42)))
    run_dir = ensure_run_dir(cfg, run_id=args.run_id)
    metrics_path, csv_path, curves_path = train_uv_probe_grid(cfg, run_dir)
    print(f"uv_probe_metrics={metrics_path}")
    print(f"uv_per_source_layer_metrics={csv_path}")
    print(f"uv_probe_training_curves={curves_path}")


if __name__ == "__main__":
    main()
