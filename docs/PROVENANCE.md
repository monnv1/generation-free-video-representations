# Artifact Provenance

Consolidated on 2026-07-15 from the following local directories:

| Consolidated path | Original source |
|---|---|
| `experiments/world_model_probe` | `/data/repos/world_model_probe` |
| `experiments/causal_denoise_probe` | `/data/repos/cosmos_causal_probe` |
| `experiments/domino_denoise_probe` | `/data/repos/domino_cosmos_denoise_probe` |
| `results/world_model_probe` | `/data/repos/world_model_probe/reports` |
| `results/causal_denoise_probe` | `/data/repos/cosmos_causal_probe/outputs` |
| `results/domino_denoise_probe` | `/data/repos/domino_cosmos_denoise_probe/results` |

The two denoise-probe source directories were not Git repositories. Their
history was reconstructed from file modification times, saved run configs, and
result metadata. `world_model_probe` was copied with its current uncommitted
working-tree changes because those changes correspond to the latest experiment
state.

## Included

- Python and shell experiment code
- YAML experiment configs
- CSV and JSON metrics
- aggregate text reports
- training curves
- compact diagnostic reports and figures
- selected HTML visualizations

## Excluded

- pretrained model weights
- DOM/DOMINO datasets
- `.pt` latent dumps and probe checkpoints
- `.npy` multi-gigabyte feature arrays
- Parquet slice caches and label archives
- videos and external repository checkouts
- W&B files, logs, virtual environments, and Python caches

The original directories remain untouched. Saved configs and JSON files may
contain absolute `/data/...` paths; these are retained as provenance and are not
secrets.

## Added Resume-Project Evidence

| Consolidated path | Original source |
|---|---|
| `integrations/starvla_domino` | `/data/repos/starVLA`, commit `db8fe59` plus the 2026-07-15 working tree |
| `results/latent_motion_probe/single_pair_metrics.json` | `first_two_future_latents_with_pred.pt` plus decoded current/future PNGs in the original causal probe |
| `results/async_domino_eval/adjust_bottle_async_summary.json` | 2026-05-17 `adjust_bottle` async evaluation log under `/data/checkpoints/StarVLA_DOMINO` |
| `results/async_domino_eval/adjust_bottle_sync_3step_summary.json` | 2026-05-24 synchronous CFG=7/sigma=1/K=3 evaluation log under `/data/checkpoints/StarVLA_DOMINO` |

The raw `.pt` tensor and multi-megabyte step logs remain excluded. The committed
JSON files are generated summaries, and the offline generation scripts live in
`tools/`.
