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
