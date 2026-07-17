# Results

This directory contains lightweight artifacts only. Large tensors, checkpoints,
datasets, latent caches, and raw evaluation logs are intentionally excluded.

## Headline evidence

- `latent_motion_probe/single_pair_metrics.json`: offline spatial association
  between a saved VAE latent delta and decoded RGB change.
- `causal_denoise_probe/outputs/save_latents_test/latent_heatmap_top5pct_overlay.png`:
  qualitative top-5% latent-change overlay for the same experiment track.
- `async_domino_eval/adjust_bottle_async_summary.json`: clean-latent async/RTC
  latency summary.
- `async_domino_eval/adjust_bottle_sync_3step_summary.json`: synchronous
  CFG=7/sigma=1/K=3 latency summary.

The two DOMINO timing summaries describe different system configurations. They
are not a matched one-variable sync/async or denoise/no-denoise ablation.

## Archived exploratory probes

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
- Native early-step and late-step UV sweeps:
  `domino_denoise_probe/domino_cosmos_uv_probe_3s_native_s_1_10_l_8_24/uv_summary.csv`
  and
  `domino_denoise_probe/domino_cosmos_uv_probe_3s_native_stop26_l8_24_s10_26/uv_summary.csv`.

Read the root README and `docs/REPRODUCIBILITY.md` before interpreting
cross-run comparisons.
