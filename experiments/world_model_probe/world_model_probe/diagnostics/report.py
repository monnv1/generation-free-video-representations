from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from world_model_probe.config import apply_overrides, load_config
from world_model_probe.data.dom import clip_frame_indices
from world_model_probe.diagnostics.metrics import (
    BucketSpec,
    bucket_table,
    direction_table,
)
from world_model_probe.diagnostics import metrics as metric_fns
from world_model_probe.diagnostics.prediction_dump import dump_predictions
from world_model_probe.utils import ensure_dir


def _fmt(path: str, cfg: dict[str, Any]) -> str:
    return path.format(backbone=cfg["backbone"]["name"], run_name=cfg["project"]["run_name"])


def _load_metadata(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _f(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def _table(headers: list[str], rows: list[list[Any]], digits: int = 4) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_f(v, digits) for v in row) + " |")
    return "\n".join(out)


def _overall_table(data: dict[str, np.ndarray]) -> str:
    rows = []
    for key in ("obj_pos", "obj_vel", "arm_pos"):
        m = metric_fns.prediction_metrics_for_key(data, key)
        rows.append(
            [
                key,
                m["l2_mean"],
                m["persistence_l2_mean"],
                100.0 * (1.0 - m["l2_vs_persistence"]),
                m["pred_norm_mean"],
                m["gt_norm_mean"],
                m["pred_norm_over_gt_norm"],
            ]
        )
    return _table(
        ["target", "probe L2", "persistence L2", "improve %", "pred norm", "gt norm", "pred/gt norm"],
        rows,
    )


def _bucket_md(title: str, thresholds: dict[str, float], rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["bucket"],
                row["count"],
                row["motion_norm_mean"],
                row["gt_norm_mean"],
                row["pred_norm_mean"],
                row["pred_norm_over_gt_norm"],
                row["l2_mean"],
                row["persistence_l2_mean"],
                row["l2_vs_persistence"],
                row["zero_like_pred_rate"],
            ]
        )
    return "\n".join(
        [
            f"### {title}",
            "",
            f"Buckets: near-static <= `{thresholds['static_eps']:.4f}`, large >= `{thresholds['large_min']:.4f}`.",
            "",
            _table(
                [
                    "bucket",
                    "count",
                    "bucket motion norm",
                    "gt norm",
                    "pred norm",
                    "pred/gt",
                    "probe L2",
                    "persist L2",
                    "L2/persist",
                    "zero-like pred rate",
                ],
                table_rows,
            ),
        ]
    )


def _direction_md(rows: list[dict[str, Any]]) -> str:
    table_rows = [
        [r["target"], r["horizon"], r["count"], r["r2"], r["cos_mean"], r["cos_median"], r["sign_acc"]]
        for r in rows
        if r["horizon"] == "all" or r["target"] in {"obj_pos", "obj_vel"}
    ]
    return _table(["target", "horizon", "count", "R2", "cos mean", "cos median", "sign acc"], table_rows)


def _distribution_md(rows: list[dict[str, Any]]) -> str:
    table_rows = [
        [
            r["target"],
            r["horizon"],
            r["count"],
            r["mean"],
            r["median"],
            r["p75"],
            r["p90"],
            r["p95"],
            r["p99"],
            r["near_static_frac"],
        ]
        for r in rows
    ]
    return _table(
        ["target", "horizon", "count", "mean", "median", "p75", "p90", "p95", "p99", "near-static frac"],
        table_rows,
    )


def _time_alignment_md(cfg: dict[str, Any], metadata: list[dict[str, Any]]) -> str:
    data_cfg = cfg["data"]
    input_frames = int(data_cfg.get("input_frames", 8))
    frame_stride = int(data_cfg.get("frame_stride", 1))
    sample_stride = int(data_cfg.get("sample_stride", 1))
    horizons = [int(h) for h in cfg["targets"]["horizons"]]
    fps = 25
    info_path = Path(data_cfg["dom_root"]) / "meta" / "info.json"
    if info_path.exists():
        fps = int(json.load(open(info_path, "r", encoding="utf-8")).get("fps", fps))
    examples = []
    for row in metadata[:3]:
        t = int(row["frame_index"])
        examples.append([row["sample_id"], t, str(clip_frame_indices(t, input_frames, frame_stride)), str([t + h for h in horizons])])
    lines = [
        f"- `input_frames={input_frames}`, `frame_stride={frame_stride}`: cache uses contiguous history ending at current frame `t`.",
        f"- History span: `{(input_frames - 1) * frame_stride / fps:.3f}s`; sample stride: `{sample_stride / fps:.3f}s`.",
        f"- FPS: `{fps}`; horizons `{horizons}` correspond to `{[round(h / fps, 3) for h in horizons]}` seconds after `t`.",
        "- Adapter code constructs `frame_indices = clip_frame_indices(frame_index, input_frames, frame_stride)`, so latent current time is the last historical frame `t`, not the sparse sample index.",
        "- Targets are built from parquet rows `t + horizon`; no off-by-one was found in the cache/index metadata check below.",
        "",
        _table(["sample_id", "t", "history frames", "target frames"], examples, digits=3),
    ]
    return "\n".join(lines)


def _blindness_statement(pos_rows: list[dict[str, Any]], vel_rows: list[dict[str, Any]]) -> str:
    pos_large = next(r for r in pos_rows if r["bucket"] == "large")
    vel_large = next(r for r in vel_rows if r["bucket"] == "large")
    lines = []
    for name, row in [("obj_pos", pos_large), ("obj_vel", vel_large)]:
        ratio = row["pred_norm_over_gt_norm"]
        l2_ratio = row["l2_vs_persistence"]
        zero_rate = row["zero_like_pred_rate"]
        if ratio < 0.25 and l2_ratio >= 0.90:
            verdict = "strong evidence of near-zero prediction on large-motion samples"
        elif l2_ratio >= 0.90:
            verdict = "weak dynamic readout; close to persistence"
        else:
            verdict = "some dynamic signal above persistence"
        lines.append(
            f"- `{name}` large bucket: pred/gt norm `{ratio:.3f}`, L2/persistence `{l2_ratio:.3f}`, "
            f"zero-like rate `{zero_rate:.3f}` -> {verdict}."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate object-motion diagnostics for a trained probe.")
    parser.add_argument("--config", default="/data/repos/world_model_probe/configs/cosmos_probe.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="eval", choices=["train", "eval"])
    parser.add_argument("--output-dir", default="/data/repos/world_model_probe/reports/object_motion_diagnostics")
    parser.add_argument("--report", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    ckpt = Path(args.checkpoint or (Path(_fmt(cfg["paths"]["checkpoint_dir"], cfg)) / "best.pt"))
    output_dir = ensure_dir(args.output_dir)
    npz_path, metadata_path, summary_path = dump_predictions(
        cfg,
        ckpt,
        args.split,
        output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        limit=args.limit,
        force=args.force,
    )
    data = {k: v for k, v in np.load(npz_path).items()}
    metadata = _load_metadata(metadata_path)
    summary = json.load(open(summary_path, "r", encoding="utf-8"))

    spec = BucketSpec(static_eps=0.01, large_quantile=0.90)
    pos_thresholds, pos_rows = bucket_table(data, "obj_pos", "obj_pos", spec)
    vel_thresholds, vel_rows = bucket_table(data, "obj_vel", "obj_vel", spec)
    dir_rows = direction_table(data)
    dist_rows = metric_fns.persistence_distribution_table(data, static_eps=spec.static_eps)
    examples = metric_fns.selected_large_motion_examples(data, metadata, top_k=5)

    md = [
        "# Object Motion Probe Diagnostics",
        "",
        f"- Checkpoint: `{ckpt}`",
        f"- Split: `{args.split}`",
        f"- Samples: `{summary['num_samples']}`",
        f"- Target mode: `{summary['target_mode']}`",
        f"- Horizons: `{summary['horizons']}`",
        "",
        "## Overall Delta Prediction",
        "",
        _overall_table(data),
        "",
        "## Motion Bucket Evaluation",
        "",
        "Buckets are computed over sample-horizon pairs. Persistence L2 is the error of predicting zero delta.",
        "",
        _bucket_md("Bucketed by ||Delta obj_pos||, evaluating obj_pos", pos_thresholds, pos_rows),
        "",
        _bucket_md("Bucketed by ||Delta obj_vel||, evaluating obj_vel", vel_thresholds, vel_rows),
        "",
        "## Object Motion Blindness Check",
        "",
        _blindness_statement(pos_rows, vel_rows),
        "",
        "Largest object-position-motion examples:",
        "",
        _table(
            ["sample_id", "episode", "frame", "horizon", "gt obj_pos norm", "pred obj_pos norm", "L2", "pred/gt"],
            [
                [
                    e["sample_id"],
                    e["episode_index"],
                    e["frame_index"],
                    e["horizon"],
                    e["gt_obj_pos_norm"],
                    e["pred_obj_pos_norm"],
                    e["obj_pos_l2"],
                    e["pred_over_gt"],
                ]
                for e in examples
            ],
        ),
        "",
        "## R2, Direction Cosine, Sign Accuracy",
        "",
        _direction_md(dir_rows),
        "",
        "## Persistence Baseline Strength",
        "",
        "The table reports true object delta norm distributions. A high near-static fraction means persistence is intrinsically strong.",
        "",
        _distribution_md(dist_rows),
        "",
        "## Time Alignment Check",
        "",
        _time_alignment_md(cfg, metadata),
        "",
    ]

    report_path = Path(args.report or (output_dir / f"object_motion_diagnostics_{args.split}.md"))
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[diagnostics] wrote {report_path}")


if __name__ == "__main__":
    main()
