from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from world_model_probe.baselines import compute_baseline_metrics
from world_model_probe.config import apply_overrides, load_config
from world_model_probe.latent_dataset import LatentProbeDataset, collate_latent_samples, infer_token_dim
from world_model_probe.model import DynamicsProbe, absolute_error_values, probe_loss, summarize_scalar_values
from world_model_probe.utils import ensure_dir, seed_everything, write_json


def _fmt(path: str, cfg: dict[str, Any]) -> str:
    return path.format(backbone=cfg["backbone"]["name"], run_name=cfg["project"]["run_name"])


def _index_path(cfg: dict[str, Any], split: str) -> Path:
    return Path(_fmt(cfg["paths"]["latent_root"], cfg)) / f"index_{split}.jsonl"


def _format_template(value: Any, cfg: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(backbone=cfg["backbone"]["name"], run_name=cfg["project"]["run_name"])
    if isinstance(value, list):
        return [_format_template(v, cfg) for v in value]
    return value


def _init_wandb(cfg: dict[str, Any], ckpt_dir: Path, input_dim: int):
    wandb_cfg = cfg.get("logging", {}).get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging is enabled, but wandb is not installed in this environment.") from exc

    mode = str(wandb_cfg.get("mode", "online"))
    run_config = dict(cfg)
    run_config["resolved"] = {
        "checkpoint_dir": str(ckpt_dir),
        "input_dim": input_dim,
    }
    try:
        return wandb.init(
            entity=_format_template(wandb_cfg.get("entity"), cfg),
            project=_format_template(wandb_cfg.get("project", "world_model_probe_DOMINO"), cfg),
            name=_format_template(wandb_cfg.get("name", cfg["project"]["run_name"]), cfg),
            group=_format_template(wandb_cfg.get("group", cfg["backbone"]["name"]), cfg),
            tags=_format_template(wandb_cfg.get("tags", []), cfg),
            mode=mode,
            resume=wandb_cfg.get("resume", "allow"),
            dir=str(ckpt_dir),
            config=run_config,
        )
    except Exception as exc:
        if mode == "online":
            raise RuntimeError(
                "wandb online init failed. Run `conda activate starVLA && wandb login`, "
                "or set `logging.wandb.mode: offline` / `logging.wandb.enabled: false`."
            ) from exc
        raise


def _build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict[str, Any], total_steps: int):
    scheduler_cfg = train_cfg.get("scheduler", {}) or {}
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type in {"none", "constant", "off", "disabled"}:
        return None
    if scheduler_type != "cosine":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))
    if total_steps <= 0:
        raise ValueError("total_steps must be positive when using a scheduler.")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate(
    model: DynamicsProbe,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    baseline_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    abs_values: dict[str, list[torch.Tensor]] = {}
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
    metrics = {k: v / max(count, 1) for k, v in totals.items()}
    metrics.update(summarize_scalar_values(abs_values))
    if baseline_metrics:
        metrics.update(baseline_metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DOM world-model dynamics probe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    seed_everything(int(cfg["project"].get("seed", 0)))
    train_cfg = cfg["training"]
    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = ensure_dir(_fmt(cfg["paths"]["checkpoint_dir"], cfg))
    shutil.copyfile(args.config, ckpt_dir / "config.yaml")

    train_ds = LatentProbeDataset(_index_path(cfg, "train"), cfg)
    eval_ds = LatentProbeDataset(_index_path(cfg, "eval"), cfg)
    train_num_workers = int(train_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True,
        collate_fn=collate_latent_samples,
        drop_last=False,
        persistent_workers=train_num_workers > 0,
    )
    eval_num_workers = int(train_cfg.get("num_workers", 4))
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 32))),
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=True,
        collate_fn=collate_latent_samples,
        persistent_workers=eval_num_workers > 0,
    )
    input_dim_cfg = cfg["probe"].get("backbone_dim", "auto")
    input_dim = infer_token_dim(_index_path(cfg, "train")) if str(input_dim_cfg).lower() == "auto" else int(input_dim_cfg)
    model = DynamicsProbe(cfg, input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 10))
    total_steps = epochs * len(train_loader)
    scheduler = _build_scheduler(optimizer, train_cfg, total_steps)
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_eval = float("inf")
    global_step = 0
    ema_beta = float(train_cfg.get("ema_beta", 0.98))
    train_ema: dict[str, float] = {}
    baseline_metrics = {}
    if bool(cfg.get("evaluation", {}).get("compute_baselines", True)):
        baseline_metrics = compute_baseline_metrics(cfg, "eval")
    wandb_run = _init_wandb(cfg, ckpt_dir, input_dim)

    try:
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                global_step += 1
                tokens = batch["tokens"].to(device, non_blocking=True)
                mask = batch["key_padding_mask"].to(device, non_blocking=True)
                targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
                valid = batch["valid"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(tokens, key_padding_mask=mask)
                    loss, metrics = probe_loss(
                        preds,
                        targets,
                        valid,
                        train_cfg.get("loss_weights", {}),
                        train_cfg.get("loss_type", "smooth_l1"),
                    )
                scaler.scale(loss).backward()
                if float(train_cfg.get("max_grad_norm", 0.0)) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                metric_values = {"loss": float(loss.item())}
                for key in model.target_keys:
                    metric_key = f"{key}_mae"
                    if metric_key in metrics:
                        metric_values[metric_key] = float(metrics[metric_key].item())
                for key, value in metric_values.items():
                    train_ema[key] = value if key not in train_ema else ema_beta * train_ema[key] + (1.0 - ema_beta) * value
                if global_step % int(train_cfg.get("log_every", 50)) == 0:
                    msg = f"[train] epoch={epoch} step={global_step} loss={float(loss.item()):.6f}"
                    for key in model.target_keys:
                        metric_key = f"{key}_mae"
                        if metric_key in metrics:
                            msg += f" {metric_key}={float(metrics[metric_key].item()):.6f}"
                    msg += f" loss_ema={train_ema['loss']:.6f}"
                    print(msg, flush=True)
                    if wandb_run is not None:
                        log_payload = {f"train/{key}": value for key, value in metric_values.items()}
                        log_payload.update({f"train/{key}_ema": value for key, value in train_ema.items()})
                        log_payload["train/epoch"] = epoch
                        log_payload["train/lr"] = optimizer.param_groups[0]["lr"]
                        wandb_run.log(log_payload, step=global_step)
                if global_step % int(train_cfg.get("eval_every", 500)) == 0:
                    eval_metrics = evaluate(model, eval_loader, device, cfg, baseline_metrics)
                    eval_loss = eval_metrics["loss"]
                    print(f"[eval] epoch={epoch} step={global_step} loss={eval_loss:.6f}", flush=True)
                    if wandb_run is not None:
                        wandb_run.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=global_step)
                    state = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "config": cfg,
                        "input_dim": input_dim,
                        "global_step": global_step,
                        "eval_metrics": eval_metrics,
                    }
                    torch.save(state, ckpt_dir / "last.pt")
                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        torch.save(state, ckpt_dir / "best.pt")
                        write_json(ckpt_dir / "best_metrics.json", eval_metrics)
                        if wandb_run is not None:
                            wandb_run.summary["best/eval_loss"] = best_eval

        eval_metrics = evaluate(model, eval_loader, device, cfg, baseline_metrics)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
            "input_dim": input_dim,
            "global_step": global_step,
            "eval_metrics": eval_metrics,
        }
        torch.save(state, ckpt_dir / "last.pt")
        if eval_metrics["loss"] <= best_eval:
            torch.save(state, ckpt_dir / "best.pt")
            write_json(ckpt_dir / "best_metrics.json", eval_metrics)
        write_json(ckpt_dir / "final_metrics.json", eval_metrics)
        if wandb_run is not None:
            wandb_run.log({f"final/{k}": v for k, v in eval_metrics.items()}, step=global_step)
            wandb_run.summary["final/loss"] = eval_metrics["loss"]
            wandb_run.summary["checkpoint_dir"] = str(ckpt_dir)
        print(f"[train] done. checkpoints -> {ckpt_dir}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
