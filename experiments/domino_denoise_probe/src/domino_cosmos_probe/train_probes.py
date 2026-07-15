from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import dump_json, ensure_run_dir, load_config, load_json, set_seed
from .probes import CurrentStateProbe, FutureDynamicsProbe, binary_metrics, masked_mae, masked_mse


def _as_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype)


def _batch_indices(indices: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator):
    order = np.array(indices, copy=True)
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def _standardize_features(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = x[train_idx].astype(np.float32)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return ((x.astype(np.float32) - mean) / std), mean.squeeze(0), std.squeeze(0)


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace("=", "_")
        .replace(".", "p")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _state_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: _state_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_state_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_state_to_cpu(value) for value in obj)
    return obj


def _current_loss(pred: dict[str, torch.Tensor], y: dict[str, torch.Tensor], weights: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xyz_loss = F.mse_loss(pred["xyz"], y["current_xyz"])
    uv_loss = masked_mse(pred["uv"], y["current_uv"], y["current_uv_valid"])
    depth_loss = F.mse_loss(pred["depth"], y["current_depth_m"])
    contact_loss, contact_acc = binary_metrics(pred["contact_logits"], y["current_contact"])
    success_loss, success_acc = binary_metrics(pred["success_logits"], y["current_success"])
    loss = (
        float(weights.get("xyz", 1.0)) * xyz_loss
        + float(weights.get("uv", 1.0)) * uv_loss
        + float(weights.get("depth", 0.2)) * depth_loss
        + float(weights.get("contact", 0.5)) * contact_loss
        + float(weights.get("success", 0.5)) * success_loss
    )
    metrics = {
        "loss": loss.detach(),
        "xyz_rmse_m": torch.sqrt(xyz_loss.detach().clamp_min(0.0)),
        "uv_mae_norm": masked_mae(pred["uv"], y["current_uv"], y["current_uv_valid"]).detach(),
        "depth_mae_m": F.l1_loss(pred["depth"], y["current_depth_m"]).detach(),
        "contact_acc": contact_acc.detach(),
        "success_acc": success_acc.detach(),
    }
    return loss, metrics


def _future_loss(pred: dict[str, torch.Tensor], y: dict[str, torch.Tensor], weights: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xyz_loss = F.mse_loss(pred["xyz"], y["future_xyz"])
    uv_loss = masked_mse(pred["uv"], y["future_uv"], y["future_uv_valid"])
    depth_loss = F.mse_loss(pred["depth"], y["future_depth_m"])
    vel_loss = F.mse_loss(pred["velocity_xyz"], y["future_velocity_xyz"])
    contact_loss, contact_acc = binary_metrics(pred["contact_logits"], y["future_contact"])
    success_loss, success_acc = binary_metrics(pred["success_logits"], y["future_success"])
    ttc_loss = F.cross_entropy(pred["time_to_contact_logits"], y["time_to_contact"])
    ttc_pred = torch.argmax(pred["time_to_contact_logits"], dim=-1)
    ttc_acc = (ttc_pred == y["time_to_contact"]).float().mean()
    ttc_mae = (ttc_pred.float() - y["time_to_contact"].float()).abs().mean()
    loss = (
        float(weights.get("xyz", 1.0)) * xyz_loss
        + float(weights.get("uv", 1.0)) * uv_loss
        + float(weights.get("depth", 0.2)) * depth_loss
        + float(weights.get("velocity_xyz", 1.0)) * vel_loss
        + float(weights.get("contact", 0.5)) * contact_loss
        + float(weights.get("success", 0.5)) * success_loss
        + float(weights.get("time_to_contact", 0.5)) * ttc_loss
    )
    metrics = {
        "loss": loss.detach(),
        "xyz_rmse_m": torch.sqrt(xyz_loss.detach().clamp_min(0.0)),
        "uv_mae_norm": masked_mae(pred["uv"], y["future_uv"], y["future_uv_valid"]).detach(),
        "depth_mae_m": F.l1_loss(pred["depth"], y["future_depth_m"]).detach(),
        "velocity_mae_m_per_frame": F.l1_loss(pred["velocity_xyz"], y["future_velocity_xyz"]).detach(),
        "contact_acc": contact_acc.detach(),
        "success_acc": success_acc.detach(),
        "time_to_contact_acc": ttc_acc.detach(),
        "time_to_contact_mae_bucket": ttc_mae.detach(),
    }
    return loss, metrics


def _prepare_labels(labels_npz, row_ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    labels = {}
    for key in labels_npz.files:
        if key == "horizons":
            continue
        arr = labels_npz[key][row_ids]
        if key == "time_to_contact":
            labels[key] = _as_tensor(arr, device, dtype=torch.long)
        else:
            labels[key] = _as_tensor(arr, device)
    return labels


def _evaluate(
    model: torch.nn.Module,
    x: torch.Tensor,
    labels: dict[str, torch.Tensor],
    indices: np.ndarray,
    batch_size: int,
    loss_fn: Callable,
    weights: dict,
) -> dict[str, float]:
    model.eval()
    accum: dict[str, float] = {}
    count = 0
    with torch.inference_mode():
        for idx in _batch_indices(indices, batch_size, shuffle=False, rng=np.random.default_rng(0)):
            pred = model(x[idx])
            _, metrics = loss_fn(pred, {k: v[idx] for k, v in labels.items()}, weights)
            b = len(idx)
            for key, value in metrics.items():
                accum[key] = accum.get(key, 0.0) + float(value.detach().cpu()) * b
            count += b
    return {key: value / max(1, count) for key, value in accum.items()}


def _train_one(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    labels: dict[str, torch.Tensor],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: dict,
    loss_fn: Callable,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float]], dict]:
    probe_cfg = cfg["probe"]
    batch_size = int(probe_cfg.get("batch_size", 128))
    epochs = int(probe_cfg.get("epochs", 40))
    patience = int(probe_cfg.get("patience", 8))
    weights = dict(probe_cfg.get("losses", {}) or {})
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
            loss, _metrics = loss_fn(pred, {k: v[idx] for k, v in labels.items()}, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += float(loss.detach().cpu()) * len(idx)
            train_count += len(idx)

        val_metrics = _evaluate(model, x, labels, val_idx, batch_size, loss_fn, weights)
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_optimizer_state = _state_to_cpu(opt.state_dict()) if save_optimizer else None
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val = _evaluate(model, x, labels, val_idx, batch_size, loss_fn, weights)
    test = _evaluate(model, x, labels, test_idx, batch_size, loss_fn, weights)
    best_info = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "model_state_dict": best_state,
        "optimizer_state_dict": best_optimizer_state,
    }
    return val, test, curves, best_info


def train_probe_grid(cfg: dict, run_dir: Path) -> tuple[Path, Path, Path]:
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
    labels = _prepare_labels(labels_npz, row_ids, device)
    split = index["split"].to_numpy()
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(f"Need non-empty train/val/test splits; got {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    probe_cfg = cfg["probe"]
    hidden_dim = int(probe_cfg.get("hidden_dim", 512))
    depth = int(probe_cfg.get("depth", 2))
    dropout = float(probe_cfg.get("dropout", 0.05))
    train_current_for_all = bool(probe_cfg.get("train_current_for_all_sources", True))
    num_horizons = int(len(labels_npz["horizons"]))
    ckpt_dir = run_dir / str(probe_cfg.get("checkpoint_dir", "probe_ckpts"))
    save_checkpoints = bool(probe_cfg.get("save_checkpoints", True))
    save_current_checkpoints = bool(probe_cfg.get("save_current_checkpoints", True))
    if save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    curve_records: list[dict] = []
    all_metrics: dict[str, dict] = {}
    for source_i, source in enumerate(tqdm(meta["sources"], desc="Probe source")):
        for layer_i, layer in enumerate(meta["layers"]):
            x_np, mean, std = _standardize_features(features[:, source_i, layer_i, :], train_idx)
            x = _as_tensor(x_np, device)
            input_dim = x.shape[-1]

            tasks_to_run = ["future"]
            if train_current_for_all or source == "raw_no_denoise":
                tasks_to_run.append("current")

            for target in tasks_to_run:
                if target == "current":
                    model = CurrentStateProbe(input_dim, hidden_dim, depth, dropout)
                    loss_fn = _current_loss
                else:
                    model = FutureDynamicsProbe(input_dim, num_horizons, hidden_dim, depth, dropout)
                    loss_fn = _future_loss
                val, test, curves, best_info = _train_one(
                    model=model,
                    x=x,
                    labels=labels,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    cfg=cfg,
                    loss_fn=loss_fn,
                    device=device,
                    rng=rng,
                )
                key = f"{target}|{source}|layer={layer}"
                ckpt_path = None
                should_save_ckpt = save_checkpoints and (target != "current" or save_current_checkpoints)
                if should_save_ckpt:
                    ckpt_path = ckpt_dir / f"{target}__{_safe_name(source)}__layer_{layer}.pt"
                    payload = {
                        "target": target,
                        "source": source,
                        "source_index": source_i,
                        "layer": layer,
                        "layer_index": layer_i,
                        "model_class": type(model).__name__,
                        "input_dim": int(input_dim),
                        "hidden_dim": hidden_dim,
                        "depth": depth,
                        "dropout": dropout,
                        "num_horizons": num_horizons if target == "future" else None,
                        "horizons": labels_npz["horizons"].astype(int).tolist(),
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
                        "target": target,
                        "source": source,
                        "layer": layer,
                    }
                    curve_record.update(curve)
                    curve_records.append(curve_record)

                all_metrics[key] = {
                    "target": target,
                    "source": source,
                    "layer": layer,
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
                        "target": target,
                        "source": source,
                        "layer": layer,
                        "split": split_name,
                    }
                    rec.update(metrics)
                    records.append(rec)

    metrics_path = run_dir / "probe_metrics.json"
    csv_path = run_dir / "per_source_layer_metrics.csv"
    curves_path = run_dir / "probe_training_curves.csv"
    dump_json(all_metrics, metrics_path)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    pd.DataFrame(curve_records).to_csv(curves_path, index=False)
    return metrics_path, csv_path, curves_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train current/future probes over cached Cosmos features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("run", {}).get("seed", 42)))
    run_dir = ensure_run_dir(cfg, run_id=args.run_id)
    metrics_path, csv_path, curves_path = train_probe_grid(cfg, run_dir)
    print(f"probe_metrics={metrics_path}")
    print(f"per_source_layer_metrics={csv_path}")
    print(f"probe_training_curves={curves_path}")


if __name__ == "__main__":
    main()
