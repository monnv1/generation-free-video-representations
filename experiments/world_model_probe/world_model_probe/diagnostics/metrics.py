from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


TARGET_KEYS = ("obj_pos", "obj_vel", "arm_pos")
OBJ_KEYS = ("obj_pos", "obj_vel")


@dataclass(frozen=True)
class BucketSpec:
    static_eps: float = 0.01
    large_quantile: float = 0.90


def norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=-1)


def valid_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    return data["valid"].astype(bool)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "var": float(values.var()),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def prediction_metrics_for_key(data: dict[str, np.ndarray], key: str, mask: np.ndarray | None = None) -> dict[str, float]:
    valid = valid_mask(data)
    if mask is not None:
        valid = valid & mask
    pred = data[f"pred_delta_{key}"]
    gt = data[f"gt_delta_{key}"]
    gt_norm = norm(gt)
    pred_norm = norm(pred)
    err = norm(pred - gt)
    out = {
        "count": int(valid.sum()),
        "gt_norm_mean": float(gt_norm[valid].mean()) if valid.any() else float("nan"),
        "pred_norm_mean": float(pred_norm[valid].mean()) if valid.any() else float("nan"),
        "l2_mean": float(err[valid].mean()) if valid.any() else float("nan"),
        "persistence_l2_mean": float(gt_norm[valid].mean()) if valid.any() else float("nan"),
        "l2_vs_persistence": float(err[valid].mean() / max(gt_norm[valid].mean(), 1e-12)) if valid.any() else float("nan"),
        "pred_norm_over_gt_norm": float(pred_norm[valid].mean() / max(gt_norm[valid].mean(), 1e-12)) if valid.any() else float("nan"),
    }
    large_gt = gt_norm > 1e-12
    zero_like = pred_norm <= 0.1 * np.maximum(gt_norm, 1e-12)
    denom = valid & large_gt
    out["zero_like_pred_rate"] = float(zero_like[denom].mean()) if denom.any() else float("nan")
    return out


def motion_bucket_thresholds(mag: np.ndarray, valid: np.ndarray, spec: BucketSpec) -> dict[str, float]:
    active = mag[valid & np.isfinite(mag)]
    dynamic = active[active > spec.static_eps]
    if dynamic.size == 0:
        large_min = spec.static_eps
    else:
        large_min = float(np.quantile(dynamic, spec.large_quantile))
    return {"static_eps": spec.static_eps, "large_min": large_min}


def bucket_masks(mag: np.ndarray, valid: np.ndarray, thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    static_eps = thresholds["static_eps"]
    large_min = thresholds["large_min"]
    return {
        "near-static": valid & (mag <= static_eps),
        "medium": valid & (mag > static_eps) & (mag < large_min),
        "large": valid & (mag >= large_min),
    }


def bucket_table(data: dict[str, np.ndarray], motion_key: str, pred_key: str, spec: BucketSpec) -> tuple[dict[str, float], list[dict[str, Any]]]:
    valid = valid_mask(data)
    mag = norm(data[f"gt_delta_{motion_key}"])
    thresholds = motion_bucket_thresholds(mag, valid, spec)
    masks = bucket_masks(mag, valid, thresholds)
    rows: list[dict[str, Any]] = []
    for bucket, mask in masks.items():
        metrics = prediction_metrics_for_key(data, pred_key, mask)
        metrics["bucket"] = bucket
        metrics["motion_key"] = motion_key
        metrics["pred_key"] = pred_key
        metrics["motion_norm_mean"] = float(mag[mask].mean()) if mask.any() else float("nan")
        rows.append(metrics)
    return thresholds, rows


def r2_score(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    values_pred = pred[mask].reshape(-1)
    values_gt = gt[mask].reshape(-1)
    if values_gt.size == 0:
        return float("nan")
    sse = float(np.square(values_pred - values_gt).sum())
    centered = values_gt - values_gt.mean()
    sst = float(np.square(centered).sum())
    return 1.0 - sse / max(sst, 1e-12)


def cosine_values(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float) -> np.ndarray:
    pred_norm = norm(pred)
    gt_norm = norm(gt)
    denom = np.maximum(pred_norm * gt_norm, eps)
    cos = (pred * gt).sum(axis=-1) / denom
    keep = mask & (gt_norm > eps)
    return cos[keep]


def sign_accuracy(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float) -> float:
    keep = mask[..., None] & (np.abs(gt) > eps)
    if not keep.any():
        return float("nan")
    return float((np.sign(pred[keep]) == np.sign(gt[keep])).mean())


def direction_table(data: dict[str, np.ndarray], eps: float = 1e-6) -> list[dict[str, Any]]:
    valid = valid_mask(data)
    horizons = data["horizons"].tolist()
    rows: list[dict[str, Any]] = []
    for key in TARGET_KEYS:
        pred = data[f"pred_delta_{key}"]
        gt = data[f"gt_delta_{key}"]
        all_cos = cosine_values(pred, gt, valid, eps)
        rows.append(
            {
                "target": key,
                "horizon": "all",
                "count": int(valid.sum()),
                "r2": r2_score(pred, gt, valid),
                "cos_mean": float(all_cos.mean()) if all_cos.size else float("nan"),
                "cos_median": float(np.median(all_cos)) if all_cos.size else float("nan"),
                "sign_acc": sign_accuracy(pred, gt, valid, eps),
            }
        )
        for hi, horizon in enumerate(horizons):
            mask = np.zeros_like(valid, dtype=bool)
            mask[:, hi] = valid[:, hi]
            cos = cosine_values(pred, gt, mask, eps)
            rows.append(
                {
                    "target": key,
                    "horizon": int(horizon),
                    "count": int(mask.sum()),
                    "r2": r2_score(pred, gt, mask),
                    "cos_mean": float(cos.mean()) if cos.size else float("nan"),
                    "cos_median": float(np.median(cos)) if cos.size else float("nan"),
                    "sign_acc": sign_accuracy(pred, gt, mask, eps),
                }
            )
    return rows


def persistence_distribution_table(data: dict[str, np.ndarray], static_eps: float = 0.01) -> list[dict[str, Any]]:
    valid = valid_mask(data)
    horizons = data["horizons"].tolist()
    rows: list[dict[str, Any]] = []
    for key in OBJ_KEYS:
        mag = norm(data[f"gt_delta_{key}"])
        for hi, horizon in enumerate(horizons):
            mask = valid[:, hi]
            values = mag[:, hi][mask]
            stats = summarize(values)
            row = {"target": key, "horizon": int(horizon), **stats}
            row["near_static_frac"] = float((values <= static_eps).mean()) if values.size else float("nan")
            rows.append(row)
    return rows


def selected_large_motion_examples(data: dict[str, np.ndarray], metadata: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    valid = valid_mask(data)
    mag = norm(data["gt_delta_obj_pos"])
    err = norm(data["pred_delta_obj_pos"] - data["gt_delta_obj_pos"])
    pred_norm = norm(data["pred_delta_obj_pos"])
    flat = []
    for i in range(mag.shape[0]):
        for h in range(mag.shape[1]):
            if valid[i, h]:
                flat.append((float(mag[i, h]), i, h, float(err[i, h]), float(pred_norm[i, h])))
    flat.sort(reverse=True)
    horizons = data["horizons"].tolist()
    out = []
    for gt_norm, i, h, l2, pred_n in flat[:top_k]:
        row = metadata[i]
        out.append(
            {
                "sample_id": row.get("sample_id"),
                "episode_index": row.get("episode_index"),
                "frame_index": row.get("frame_index"),
                "horizon": int(horizons[h]),
                "gt_obj_pos_norm": gt_norm,
                "pred_obj_pos_norm": pred_n,
                "obj_pos_l2": l2,
                "pred_over_gt": pred_n / max(gt_norm, 1e-12),
            }
        )
    return out
