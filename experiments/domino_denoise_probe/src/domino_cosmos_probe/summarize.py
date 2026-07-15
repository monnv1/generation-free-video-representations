from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import dump_json


def summarize_run(run_dir: Path) -> tuple[Path, Path]:
    metrics_csv = run_dir / "per_source_layer_metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_csv}")
    df = pd.read_csv(metrics_csv)
    test = df[df["split"] == "test"].copy()

    records: list[dict] = []
    future = test[test["target"] == "future"]
    for layer in sorted(future["layer"].unique()):
        layer_df = future[future["layer"] == layer]
        raw_rows = layer_df[layer_df["source"] == "raw_no_denoise"]
        if raw_rows.empty:
            continue
        raw = raw_rows.iloc[0]
        raw_loss = float(raw["loss"])
        raw_xyz = float(raw.get("xyz_rmse_m", float("nan")))
        raw_ttc_acc = float(raw.get("time_to_contact_acc", float("nan")))
        for _, row in layer_df.iterrows():
            loss = float(row["loss"])
            rec = {
                "layer": int(layer),
                "source": row["source"],
                "target": "future",
                "test_loss": loss,
                "loss_improvement_vs_raw": (raw_loss - loss) / raw_loss if raw_loss > 0 else float("nan"),
                "xyz_rmse_m": float(row.get("xyz_rmse_m", float("nan"))),
                "xyz_rmse_improvement_vs_raw": (raw_xyz - float(row.get("xyz_rmse_m", raw_xyz))) / raw_xyz
                if raw_xyz > 0
                else float("nan"),
                "time_to_contact_acc": float(row.get("time_to_contact_acc", float("nan"))),
                "time_to_contact_acc_delta_vs_raw": float(row.get("time_to_contact_acc", raw_ttc_acc)) - raw_ttc_acc,
                "contact_acc": float(row.get("contact_acc", float("nan"))),
                "success_acc": float(row.get("success_acc", float("nan"))),
            }
            records.append(rec)

    current = test[test["target"] == "current"].copy()
    current_raw = current[current["source"] == "raw_no_denoise"].copy()
    sanity = []
    for _, row in current_raw.iterrows():
        sanity.append(
            {
                "layer": int(row["layer"]),
                "source": row["source"],
                "target": "current_sanity",
                "test_loss": float(row["loss"]),
                "xyz_rmse_m": float(row.get("xyz_rmse_m", float("nan"))),
                "uv_mae_norm": float(row.get("uv_mae_norm", float("nan"))),
                "contact_acc": float(row.get("contact_acc", float("nan"))),
                "success_acc": float(row.get("success_acc", float("nan"))),
            }
        )

    summary = pd.DataFrame(records).sort_values(
        ["layer", "loss_improvement_vs_raw"],
        ascending=[True, False],
    )
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    best = {}
    if not summary.empty:
        best_row = summary[summary["source"] != "raw_no_denoise"].sort_values("loss_improvement_vs_raw", ascending=False).head(1)
        if not best_row.empty:
            best = best_row.iloc[0].to_dict()
    report = {
        "summary_csv": str(summary_path),
        "metrics_csv": str(metrics_csv),
        "training_curves_csv": str(run_dir / "probe_training_curves.csv"),
        "checkpoint_dir": str(run_dir / "probe_ckpts"),
        "best_future_denoise_vs_raw": best,
        "raw_current_sanity": sanity,
    }
    report_path = run_dir / "summary.json"
    dump_json(report, report_path)
    return summary_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize denoising probe results.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    summary_path, report_path = summarize_run(Path(args.run_dir))
    print(f"summary_csv={summary_path}")
    print(f"summary_json={report_path}")


if __name__ == "__main__":
    main()
