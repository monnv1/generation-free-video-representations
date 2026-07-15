# Results

This directory contains lightweight artifacts only. Large tensors, model
checkpoints, datasets, and latent caches are intentionally excluded.

## Recommended entry points

- Frozen-token diagnostics:
  `world_model_probe/object_motion_diagnostics/object_motion_diagnostics_eval.md`
- History/sigma/step causal sweep:
  `causal_denoise_probe/outputs/history_sigma_steps_sweep/aggregate_report.txt`
- CFG and multi-slot sweep:
  `causal_denoise_probe/outputs/cfg_sequence_sweep_small/aggregate_report.txt`
- Initial multi-target DOMINO probe:
  `domino_denoise_probe/domino_cosmos_denoise_probe_20260525_144759/summary.csv`
- Corrected first UV-only metrics:
  `domino_denoise_probe/domino_cosmos_uv_probe_20260526_005100/uv_valid_weighted_metrics_recomputed.csv`
- Native early-step UV sweep:
  `domino_denoise_probe/domino_cosmos_uv_probe_3s_native_s_1_10_l_8_24/uv_summary.csv`
- Native late-step UV sweep:
  `domino_denoise_probe/domino_cosmos_uv_probe_3s_native_stop26_l8_24_s10_26/uv_summary.csv`

See the root README and `docs/EXPERIMENT_SUMMARY_ZH.md` before interpreting
cross-run comparisons.
