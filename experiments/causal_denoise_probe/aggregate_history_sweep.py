#!/usr/bin/env python3
"""Aggregate history/sigma/denoise-step sweep outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / 'summary.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def enrich_rows(rows: list[dict[str, str]], run_dir: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = summary.get('config', {}) if summary else {}
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item['run_dir'] = str(run_dir)
        item['cond_frames'] = cfg.get('cond_frames', '')
        item['future_pixel_frames'] = cfg.get('future_pixel_frames', '')
        item['frame_start'] = cfg.get('frame_start', '')
        item['frame_stride'] = cfg.get('frame_stride', '')
        item['cond_latent_frames'] = summary.get('cond_latent_frames', '')
        item['summary_target_latent_idx'] = summary.get('target_latent_idx', '')
        item['latent_shape'] = 'x'.join(str(x) for x in summary.get('latent_shape', []))
        enriched.append(item)
    return enriched


def as_float(row: dict[str, Any], key: str, default: float = float('-inf')) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())

    all_a1: list[dict[str, Any]] = []
    all_a0: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = load_summary(run_dir)
        all_a1.extend(enrich_rows(read_csv(run_dir / 'a1_results.csv'), run_dir, summary))
        all_a0.extend(enrich_rows(read_csv(run_dir / 'a0_results.csv'), run_dir, summary))

    write_csv(output_root / 'combined_a1_results.csv', all_a1)
    write_csv(output_root / 'combined_a0_results.csv', all_a0)

    best_by_history: list[dict[str, Any]] = []
    cond_values = sorted({str(row.get('cond_frames', '')) for row in all_a1}, key=lambda x: int(x) if x.isdigit() else -1)
    for cond in cond_values:
        subset = [row for row in all_a1 if str(row.get('cond_frames', '')) == cond]
        if not subset:
            continue
        best = max(subset, key=lambda row: as_float(row, 'cos_x_final_vs_baseline_diff'))
        best_by_history.append(best)
    write_csv(output_root / 'best_a1_by_history.csv', best_by_history)

    best_overall = max(all_a1, key=lambda row: as_float(row, 'cos_x_final_vs_baseline_diff')) if all_a1 else None
    report_lines = []
    report_lines.append(f'output_root: {output_root}')
    report_lines.append(f'runs: {len(run_dirs)}')
    report_lines.append(f'A1 rows: {len(all_a1)}')
    report_lines.append(f'A0 rows: {len(all_a0)}')
    report_lines.append('')
    report_lines.append('Best A1 by history:')
    for row in best_by_history:
        report_lines.append(
            '  cond_frames={cond} cond_lat={clat} future_slots={slots} cfg={cfg} sigma={sigma} K={k} '
            'x_final={xcos:.4f} gain={gain:+.4f} seq={seq:.4f} seq_gain={seqgain:+.4f} pred={pcos:.4f} hidden={hcos:.4f}'.format(
                cond=row.get('cond_frames', ''),
                clat=row.get('cond_latent_frames', ''),
                slots=row.get('future_latent_slots', ''),
                cfg=row.get('guidance_scale', ''),
                sigma=row.get('sigma_start', ''),
                k=row.get('denoise_steps', ''),
                xcos=as_float(row, 'latent_cos_x_final', 0.0),
                gain=as_float(row, 'cos_x_final_vs_baseline_diff', 0.0),
                seq=as_float(row, 'sequence_cos_x_final', 0.0),
                seqgain=as_float(row, 'sequence_cos_x_final_vs_baseline_diff', 0.0),
                pcos=as_float(row, 'latent_cos_pred_x0', 0.0),
                hcos=as_float(row, 'hidden_cos', 0.0),
            )
        )
    if best_overall is not None:
        report_lines.append('')
        report_lines.append(
            'Best overall: cond_frames={cond} future_slots={slots} cfg={cfg} sigma={sigma} K={k} '
            'gain={gain:+.4f} x_final={xcos:.4f} seq_gain={seqgain:+.4f}'.format(
                cond=best_overall.get('cond_frames', ''),
                slots=best_overall.get('future_latent_slots', ''),
                cfg=best_overall.get('guidance_scale', ''),
                sigma=best_overall.get('sigma_start', ''),
                k=best_overall.get('denoise_steps', ''),
                gain=as_float(best_overall, 'cos_x_final_vs_baseline_diff', 0.0),
                xcos=as_float(best_overall, 'latent_cos_x_final', 0.0),
                seqgain=as_float(best_overall, 'sequence_cos_x_final_vs_baseline_diff', 0.0),
            )
        )

    text = '\n'.join(report_lines) + '\n'
    (output_root / 'aggregate_report.txt').write_text(text, encoding='utf-8')
    print(text)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_root / 'aggregate_plot_skipped.txt').write_text(f'matplotlib import failed: {exc}\n', encoding='utf-8')
        return

    if best_by_history:
        xs = [int(row['cond_frames']) for row in best_by_history]
        gains = [as_float(row, 'cos_x_final_vs_baseline_diff', 0.0) for row in best_by_history]
        xfinal = [as_float(row, 'latent_cos_x_final', 0.0) for row in best_by_history]
        baseline = [as_float(row, 'prev_latent_baseline_cos', 0.0) for row in best_by_history]

        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(xs, gains, marker='o', label='best gain over baseline')
        ax1.axhline(0.0, linestyle='--', color='black', linewidth=1)
        ax1.set_xlabel('condition pixel frames')
        ax1.set_ylabel('best cos_x_final_vs_baseline_diff')
        ax1.set_title('Best A1 gain by history length')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.savefig(output_root / 'best_gain_by_history.png', dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, xfinal, marker='o', label='best x_final cosine')
        ax.plot(xs, baseline, marker='x', label='previous latent baseline')
        ax.set_xlabel('condition pixel frames')
        ax.set_ylabel('cosine to target future latent')
        ax.set_title('Best A1 x_final vs baseline by history length')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(output_root / 'best_xfinal_vs_baseline_by_history.png', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()
