from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TARGET_KEYS = ("obj_pos", "obj_vel", "arm_pos")


def target_keys_from_config(cfg: dict[str, Any]) -> tuple[str, ...]:
    target_cfg = cfg.get("targets", {})
    raw_keys = target_cfg.get("target_keys", target_cfg.get("keys", DEFAULT_TARGET_KEYS))
    if isinstance(raw_keys, str):
        keys = (raw_keys,)
    else:
        keys = tuple(str(key) for key in raw_keys)
    if not keys:
        raise ValueError("targets.target_keys must contain at least one target.")
    unknown = sorted(set(keys) - set(DEFAULT_TARGET_KEYS))
    if unknown:
        raise ValueError(f"Unsupported target key(s): {unknown}. Supported keys: {list(DEFAULT_TARGET_KEYS)}")
    return keys


def target_dim_from_config(target_cfg: dict[str, Any], key: str) -> int:
    if key == "obj_pos":
        return len(target_cfg.get("obj_pos_indices", [0, 1, 2]))
    if key == "obj_vel":
        return len(target_cfg.get("obj_vel_indices", [6, 7, 8]))
    if key == "arm_pos":
        return len(target_cfg.get("arm_pos_indices", [0, 1, 2]))
    raise ValueError(f"Unsupported target key: {key}")


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReadoutBlock(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        use_self_attn: bool = True,
    ) -> None:
        super().__init__()
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_norm = nn.LayerNorm(query_dim)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=query_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.cross_norm = nn.LayerNorm(query_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, query_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_self_attn:
            q = self.self_norm(queries)
            queries = queries + self.dropout(
                self.self_attn(
                    query=q,
                    key=q,
                    value=q,
                    need_weights=False,
                )[0]
            )
        q = self.cross_norm(queries)
        queries = queries + self.dropout(
            self.cross_attn(
                query=q,
                key=memory,
                value=memory,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        )
        queries = queries + self.dropout(self.ffn(self.ffn_norm(queries)))
        return queries


class DynamicsProbe(nn.Module):
    """Cross-attention probe for frozen world-model tokens."""

    def __init__(self, cfg: dict[str, Any], input_dim: int) -> None:
        super().__init__()
        pcfg = cfg["probe"]
        target_cfg = cfg["targets"]
        self.target_keys = target_keys_from_config(cfg)
        self.horizons = [int(h) for h in target_cfg["horizons"]]
        self.num_horizons = len(self.horizons)
        self.query_dim = int(pcfg.get("query_dim", 512))
        self.num_queries = int(pcfg.get("num_queries", 16))
        self.num_obj_queries = int(pcfg.get("num_obj_queries", self.num_queries // 2))
        if self.num_obj_queries <= 0 or self.num_obj_queries >= self.num_queries:
            raise ValueError("probe.num_obj_queries must be in [1, num_queries - 1].")
        num_heads = int(pcfg.get("num_heads", 8))
        dropout = float(pcfg.get("dropout", 0.0))
        head_hidden_dim = int(pcfg.get("head_hidden_dim", self.query_dim))
        self.num_readout_layers = int(pcfg.get("num_readout_layers", pcfg.get("num_layers", 1)))
        if self.num_readout_layers < 1:
            raise ValueError("probe.num_readout_layers must be >= 1.")
        self.use_readout_blocks = bool(pcfg.get("use_readout_blocks", self.num_readout_layers > 1))
        readout_ffn_dim = int(pcfg.get("readout_ffn_dim", self.query_dim * 4))
        use_self_attn = bool(pcfg.get("use_query_self_attn", True))
        self.target_dims = {
            key: target_dim_from_config(target_cfg, key)
            for key in self.target_keys
        }

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, self.query_dim)
        self.queries = nn.Parameter(torch.randn(self.num_queries, self.query_dim) * 0.02)
        if self.use_readout_blocks:
            self.readout_blocks = nn.ModuleList(
                [
                    ReadoutBlock(
                        query_dim=self.query_dim,
                        num_heads=num_heads,
                        ffn_dim=readout_ffn_dim,
                        dropout=dropout,
                        use_self_attn=use_self_attn,
                    )
                    for _ in range(self.num_readout_layers)
                ]
            )
            self.cross_attn = None
        else:
            self.readout_blocks = nn.ModuleList()
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.query_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.query_norm = nn.LayerNorm(self.query_dim)
        merged_dim = self.query_dim * 2
        if "obj_pos" in self.target_keys:
            self.obj_pos_head = MLPHead(merged_dim, head_hidden_dim, self.num_horizons * self.target_dims["obj_pos"], dropout)
        if "obj_vel" in self.target_keys:
            self.obj_vel_head = MLPHead(merged_dim, head_hidden_dim, self.num_horizons * self.target_dims["obj_vel"], dropout)
        if "arm_pos" in self.target_keys:
            self.arm_pos_head = MLPHead(merged_dim, head_hidden_dim, self.num_horizons * self.target_dims["arm_pos"], dropout)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if tokens.dim() != 3:
            raise ValueError(f"Expected tokens [B,N,D], got {tuple(tokens.shape)}")
        x = self.input_proj(self.input_norm(tokens))
        batch_size = x.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        if self.use_readout_blocks:
            attended = queries
            for block in self.readout_blocks:
                attended = block(attended, x, key_padding_mask=key_padding_mask)
        else:
            if self.cross_attn is None:
                raise RuntimeError("DynamicsProbe legacy cross_attn is not initialized.")
            attended, _ = self.cross_attn(
                query=queries,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        attended = self.query_norm(attended)
        obj_feat = attended[:, : self.num_obj_queries].mean(dim=1)
        arm_feat = attended[:, self.num_obj_queries :].mean(dim=1)
        shared = attended.mean(dim=1)
        obj_out = torch.cat([obj_feat, shared], dim=-1)
        arm_out = torch.cat([arm_feat, shared], dim=-1)
        outputs: dict[str, torch.Tensor] = {}
        if "obj_pos" in self.target_keys:
            outputs["obj_pos"] = self.obj_pos_head(obj_out).view(batch_size, self.num_horizons, self.target_dims["obj_pos"])
        if "obj_vel" in self.target_keys:
            outputs["obj_vel"] = self.obj_vel_head(obj_out).view(batch_size, self.num_horizons, self.target_dims["obj_vel"])
        if "arm_pos" in self.target_keys:
            outputs["arm_pos"] = self.arm_pos_head(arm_out).view(batch_size, self.num_horizons, self.target_dims["arm_pos"])
        return outputs


def probe_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    valid: torch.Tensor,
    weights: dict[str, float],
    loss_type: str = "smooth_l1",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not preds:
        raise ValueError("probe_loss requires at least one prediction target.")
    valid = valid.to(next(iter(preds.values())).dtype)
    total = torch.zeros((), device=valid.device, dtype=valid.dtype)
    metrics: dict[str, torch.Tensor] = {}
    for key, pred in preds.items():
        if key not in targets:
            raise KeyError(f"Missing target for prediction key {key!r}.")
        target = targets[key].to(device=pred.device, dtype=pred.dtype)
        loss_kind = str(loss_type).lower()
        if loss_kind == "mse":
            err = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        elif loss_kind == "l2":
            err = torch.linalg.vector_norm(pred - target, dim=-1)
        elif loss_kind == "smooth_l1":
            err = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        denom = valid.sum().clamp_min(1.0)
        loss = (err * valid).sum() / denom
        mae = (torch.abs(pred - target).mean(dim=-1) * valid).sum() / denom
        l2 = (torch.linalg.vector_norm(pred - target, dim=-1) * valid).sum() / denom
        weight = float(weights.get(key, 1.0))
        total = total + weight * loss
        metrics[f"{key}_loss"] = loss.detach()
        metrics[f"{key}_mae"] = mae.detach()
        metrics[f"{key}_l2"] = l2.detach()
        with torch.no_grad():
            per_h = (torch.abs(pred - target).mean(dim=-1) * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1.0)
            for i, value in enumerate(per_h):
                metrics[f"{key}_mae_h{int(i)}"] = value.detach()
            per_h_l2 = (torch.linalg.vector_norm(pred - target, dim=-1) * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1.0)
            for i, value in enumerate(per_h_l2):
                metrics[f"{key}_l2_h{int(i)}"] = value.detach()
    metrics["loss"] = total.detach()
    return total, metrics


def absolute_error_values(
    preds: dict[str, torch.Tensor],
    absolute_targets: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    valid: torch.Tensor,
    target_mode: str,
    relative_eps: float = 1.0e-6,
    keys: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    if not preds or not absolute_targets:
        return {}

    target_mode = target_mode.lower()
    valid_mask = valid.to(device=next(iter(preds.values())).device) > 0
    out: dict[str, torch.Tensor] = {}
    keys = keys or tuple(preds.keys())

    arm_disp = None
    if "arm_pos" in current_state and "arm_pos" in absolute_targets:
        arm_current = current_state["arm_pos"].to(device=valid_mask.device, dtype=next(iter(preds.values())).dtype)
        arm_target = absolute_targets["arm_pos"].to(device=valid_mask.device, dtype=arm_current.dtype)
        arm_disp = torch.linalg.vector_norm(arm_target - arm_current.unsqueeze(1), dim=-1)
        out["arm_disp"] = arm_disp[valid_mask].detach().float().cpu()

    for key in keys:
        if key not in preds or key not in absolute_targets:
            continue
        pred = preds[key]
        target = absolute_targets[key].to(device=pred.device, dtype=pred.dtype)
        if target_mode == "delta":
            if key not in current_state:
                continue
            current = current_state[key].to(device=pred.device, dtype=pred.dtype)
            pred_abs = pred + current.unsqueeze(1)
        else:
            pred_abs = pred
        err_l2 = torch.linalg.vector_norm(pred_abs - target, dim=-1)
        out[f"{key}_abs_l2"] = err_l2[valid_mask].detach().float().cpu()
        if arm_disp is not None:
            rel = err_l2 / arm_disp.to(device=pred.device, dtype=pred.dtype).clamp_min(relative_eps)
            out[f"{key}_abs_rel_arm_disp"] = rel[valid_mask].detach().float().cpu()
    return {k: v for k, v in out.items() if v.numel() > 0}


def summarize_scalar_values(values: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, chunks in values.items():
        if not chunks:
            continue
        x = torch.cat(chunks).float()
        if x.numel() == 0:
            continue
        metrics[f"{name}_mean"] = float(x.mean().item())
        metrics[f"{name}_median"] = float(x.median().item())
        metrics[f"{name}_min"] = float(x.min().item())
        metrics[f"{name}_max"] = float(x.max().item())
        metrics[f"{name}_var"] = float(x.var(unbiased=False).item())
    return metrics
