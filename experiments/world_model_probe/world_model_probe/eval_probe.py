from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from world_model_probe.baselines import compute_baseline_metrics
from world_model_probe.config import apply_overrides, load_config
from world_model_probe.latent_dataset import LatentProbeDataset, collate_latent_samples
from world_model_probe.model import DynamicsProbe, absolute_error_values, probe_loss, summarize_scalar_values, target_keys_from_config
from world_model_probe.utils import ensure_dir, write_json


def _csv_columns(target_keys: tuple[str, ...]) -> list[str]:
    return ["index"] + [f"{key}_{kind}" for key in target_keys for kind in ("true", "pred")]


def _fmt(path: str, cfg: dict[str, Any], **extra: Any) -> str:
    fields = {"backbone": cfg["backbone"]["name"], "run_name": cfg["project"]["run_name"]}
    fields.update(extra)
    return path.format(**fields)


def _index_path(cfg: dict[str, Any], split: str) -> Path:
    return Path(_fmt(cfg["paths"]["latent_root"], cfg)) / f"index_{split}.jsonl"


def _prediction_csv_path(cfg: dict[str, Any], split: str, output_path: Path) -> Path:
    configured = cfg.get("evaluation", {}).get("prediction_csv")
    if configured:
        return Path(_fmt(str(configured), cfg, split=split))
    return output_path.with_suffix(".csv")


def _tensor_cell(value: torch.Tensor) -> str:
    flat = value.detach().float().cpu().reshape(-1).tolist()
    return json.dumps([float(x) for x in flat], separators=(",", ":"))


def _absolute_prediction_for_key(
    key: str,
    preds: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    target_mode: str,
) -> torch.Tensor:
    pred = preds[key]
    if target_mode.lower() != "delta":
        return pred
    if key not in current_state:
        raise ValueError(f"Cannot convert delta prediction for {key!r} without current_state.")
    return pred + current_state[key].to(device=pred.device, dtype=pred.dtype).unsqueeze(1)


def _append_prediction_rows(
    rows: list[dict[str, str]],
    batch: dict[str, Any],
    preds: dict[str, torch.Tensor],
    target_mode: str,
    target_keys: tuple[str, ...],
) -> None:
    absolute_targets = batch["absolute_targets"]
    current_state = batch["current_state"]
    valid = batch["valid"].detach().cpu() > 0
    pred_abs = {
        key: _absolute_prediction_for_key(key, preds, current_state, target_mode).detach().cpu()
        for key in target_keys
    }
    target_abs = {key: absolute_targets[key].detach().cpu() for key in target_keys}
    batch_size = next(iter(pred_abs.values())).shape[0]
    num_horizons = next(iter(pred_abs.values())).shape[1]
    for bi in range(batch_size):
        for hi in range(num_horizons):
            if not bool(valid[bi, hi]):
                continue
            row = {"index": str(len(rows))}
            for key in target_keys:
                row[f"{key}_true"] = _tensor_cell(target_abs[key][bi, hi])
                row[f"{key}_pred"] = _tensor_cell(pred_abs[key][bi, hi])
            rows.append(row)


def _write_prediction_csv(path: Path, rows: list[dict[str, str]], target_keys: tuple[str, ...]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_columns(target_keys))
        writer.writeheader()
        writer.writerows(rows)


def _metrics_from_prediction_csv(path: Path, target_keys: tuple[str, ...]) -> dict[str, float]:
    diffs: dict[str, list[np.ndarray]] = {key: [] for key in target_keys}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in target_keys:
                true = np.asarray(json.loads(row[f"{key}_true"]), dtype=np.float64)
                pred = np.asarray(json.loads(row[f"{key}_pred"]), dtype=np.float64)
                diffs[key].append(pred - true)

    metrics: dict[str, float] = {}
    total_l2: list[np.ndarray] = []
    for key, chunks in diffs.items():
        if not chunks:
            continue
        arr = np.stack(chunks, axis=0)
        abs_arr = np.abs(arr)
        l2 = np.linalg.norm(arr, axis=1)
        total_l2.append(l2)
        metrics[f"csv/{key}_mae"] = float(abs_arr.mean())
        metrics[f"csv/{key}_rmse"] = float(np.sqrt(np.square(arr).mean()))
        metrics[f"csv/{key}_l2_mean"] = float(l2.mean())
        metrics[f"csv/{key}_l2_median"] = float(np.median(l2))
        metrics[f"csv/{key}_l2_min"] = float(l2.min())
        metrics[f"csv/{key}_l2_max"] = float(l2.max())
        metrics[f"csv/{key}_l2_var"] = float(l2.var())
    if total_l2:
        merged = np.concatenate(total_l2, axis=0)
        metrics["csv/all_l2_mean"] = float(merged.mean())
        metrics["csv/all_l2_median"] = float(np.median(merged))
    metrics["csv/num_rows"] = float(sum(len(v) for v in diffs.values()) / max(len(target_keys), 1))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the DOM world-model dynamics probe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="eval", choices=["train", "eval"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--prediction-csv", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    target_keys = target_keys_from_config(cfg)
    device = torch.device(cfg["evaluation"].get("device", cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    ckpt_path = Path(args.checkpoint or (Path(_fmt(cfg["paths"]["checkpoint_dir"], cfg)) / "best.pt"))
    state = torch.load(ckpt_path, map_location="cpu")
    input_dim = int(state.get("input_dim", cfg["probe"].get("backbone_dim")))
    model = DynamicsProbe(cfg, input_dim=input_dim).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    ds = LatentProbeDataset(_index_path(cfg, args.split), cfg)
    num_workers = int(cfg["evaluation"].get("num_workers", cfg["training"].get("num_workers", 4)))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["evaluation"].get("batch_size", cfg["training"].get("eval_batch_size", 64))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_latent_samples,
        persistent_workers=num_workers > 0,
    )
    totals: dict[str, float] = {}
    abs_values: dict[str, list[torch.Tensor]] = {}
    prediction_rows: list[dict[str, str]] = []
    count = 0
    target_mode = str(cfg.get("targets", {}).get("mode", "absolute"))
    relative_eps = float(cfg.get("evaluation", {}).get("relative_error_eps", 1.0e-6))
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            mask = batch["key_padding_mask"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            absolute_targets = {k: v.to(device) for k, v in batch["absolute_targets"].items()}
            current_state = {k: v.to(device) for k, v in batch["current_state"].items()}
            valid = batch["valid"].to(device)
            preds = model(tokens, key_padding_mask=mask)
            _, metrics = probe_loss(
                preds,
                targets,
                valid,
                cfg["training"].get("loss_weights", {}),
                cfg["training"].get("loss_type", "smooth_l1"),
            )
            bs = tokens.shape[0]
            count += bs
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v.item()) * bs
            for k, v in absolute_error_values(
                preds,
                absolute_targets,
                current_state,
                valid,
                target_mode,
                relative_eps,
            ).items():
                abs_values.setdefault(k, []).append(v)
            csv_batch = dict(batch)
            csv_batch["absolute_targets"] = absolute_targets
            csv_batch["current_state"] = current_state
            csv_batch["valid"] = valid
            _append_prediction_rows(prediction_rows, csv_batch, preds, target_mode, target_keys)

    metrics = {k: v / max(count, 1) for k, v in totals.items()}
    metrics.update(summarize_scalar_values(abs_values))
    if bool(cfg.get("evaluation", {}).get("compute_baselines", True)):
        metrics.update(compute_baseline_metrics(cfg, args.split))
    metrics["num_samples"] = count
    metrics["checkpoint"] = str(ckpt_path)
    metrics["split"] = args.split
    output_path = Path(args.output or (Path(_fmt(cfg["paths"]["checkpoint_dir"], cfg)) / f"eval_{args.split}.json"))
    csv_path = Path(args.prediction_csv) if args.prediction_csv else _prediction_csv_path(cfg, args.split, output_path)
    ensure_dir(output_path.parent)
    _write_prediction_csv(csv_path, prediction_rows, target_keys)
    metrics.update(_metrics_from_prediction_csv(csv_path, target_keys))
    metrics["prediction_csv"] = str(csv_path)
    write_json(output_path, metrics)
    print(f"[eval] wrote {output_path}", flush=True)
    print(f"[eval] wrote {csv_path}", flush=True)
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
