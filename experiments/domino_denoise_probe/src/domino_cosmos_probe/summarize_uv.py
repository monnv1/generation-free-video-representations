from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import dump_json


def summarize_uv_run(run_dir: Path) -> tuple[Path, Path]:
    metrics_csv = run_dir / "uv_per_source_layer_metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing UV metrics CSV: {metrics_csv}")
    df = pd.read_csv(metrics_csv)
    test = df[df["split"] == "test"].copy()

    records: list[dict] = []
    future = test[test["target"] == "future_uv"]
    group_cols = ["layer"]
    if "horizon" in future.columns and future["horizon"].notna().any():
        group_cols.append("horizon")
    if "future_step" in future.columns and future["future_step"].notna().any():
        group_cols.append("future_step")
    for group_key, layer_df in future.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group = dict(zip(group_cols, group_key))
        raw_rows = layer_df[layer_df["source"] == "raw_no_denoise"]
        if raw_rows.empty:
            continue
        raw = raw_rows.iloc[0]
        raw_loss = float(raw["loss"])
        raw_mae_px = float(raw.get("uv_mae_px", float("nan")))
        for _, row in layer_df.iterrows():
            loss = float(row["loss"])
            mae_px = float(row.get("uv_mae_px", float("nan")))
            record = {
                "layer": int(row["layer"]),
                "source": row["source"],
                "target": "future_uv",
                "test_loss": loss,
                "loss_improvement_vs_raw": (raw_loss - loss) / raw_loss if raw_loss > 0 else float("nan"),
                "uv_mse_norm": float(row.get("uv_mse_norm", float("nan"))),
                "uv_rmse_norm": float(row.get("uv_rmse_norm", float("nan"))),
                "uv_mae_norm": float(row.get("uv_mae_norm", float("nan"))),
                "uv_mae_px": mae_px,
                "uv_mae_px_improvement_vs_raw": (raw_mae_px - mae_px) / raw_mae_px
                if raw_mae_px > 0
                else float("nan"),
                "valid_fraction": float(row.get("valid_fraction", float("nan"))),
            }
            if "horizon" in group:
                record["horizon"] = int(group["horizon"])
            if "future_step" in group:
                record["future_step"] = int(group["future_step"])
                record["future_step_1based"] = int(group["future_step"]) + 1
            records.append(record)

    current = test[test["target"] == "current_uv"].copy()
    current_raw = current[current["source"] == "raw_no_denoise"].copy()
    sanity = []
    for _, row in current_raw.iterrows():
        sanity.append(
            {
                "layer": int(row["layer"]),
                "source": row["source"],
                "target": "current_uv_sanity",
                "test_loss": float(row["loss"]),
                "uv_mae_norm": float(row.get("uv_mae_norm", float("nan"))),
                "uv_mae_px": float(row.get("uv_mae_px", float("nan"))),
                "valid_fraction": float(row.get("valid_fraction", float("nan"))),
            }
        )

    summary = pd.DataFrame(records)
    if not summary.empty:
        sort_cols = [col for col in ["layer", "horizon", "loss_improvement_vs_raw"] if col in summary.columns]
        ascending = [True] * (len(sort_cols) - 1) + [False]
        summary = summary.sort_values(sort_cols, ascending=ascending)
    summary_path = run_dir / "uv_summary.csv"
    summary.to_csv(summary_path, index=False)

    best = {}
    if not summary.empty:
        best_row = summary[summary["source"] != "raw_no_denoise"].sort_values("loss_improvement_vs_raw", ascending=False).head(1)
        if not best_row.empty:
            best = best_row.iloc[0].to_dict()

    report = {
        "uv_summary_csv": str(summary_path),
        "uv_metrics_csv": str(metrics_csv),
        "uv_training_curves_csv": str(run_dir / "uv_probe_training_curves.csv"),
        "uv_checkpoint_dir": str(run_dir / "uv_probe_ckpts"),
        "best_future_uv_denoise_vs_raw": best,
        "raw_current_uv_sanity": sanity,
    }
    report_path = run_dir / "uv_summary.json"
    dump_json(report, report_path)
    return summary_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize UV-only denoising probe results.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    summary_path, report_path = summarize_uv_run(Path(args.run_dir))
    print(f"uv_summary_csv={summary_path}")
    print(f"uv_summary_json={report_path}")


if __name__ == "__main__":
    main()
