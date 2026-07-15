# UV-Only Cosmos Denoise Probe

This is a separate, renamed experiment path from the earlier multi-target
probe. It trains probes that predict only camera-space keypoint UV positions.
No xyz, depth, velocity, contact, success, or time-to-contact targets are used.

## Entry Point

```bash
bash scripts/run_uv_experiment.sh configs/uv_only.yaml
```

The default run id prefix is:

```text
domino_cosmos_uv_probe_<timestamp>
```

## Probe Types

The source/layer grid is unchanged:

```text
source = raw_no_denoise + 8 denoise tau values
layer = 6, 14, 27
```

Two UV-only targets are trained for each source/layer:

```text
current_uv: h_{source,l} -> uv[t]
future_uv:  h_{source,l} -> uv[t+1,t+2,t+4,t+8,t+15]
```

With the default grid this is:

```text
9 sources x 3 layers x 2 UV targets = 54 UV-only probes
```

## Loss

The only loss is masked normalized UV MSE:

```text
L_uv = masked_MSE(pred_uv, target_uv)
```

The mask is:

```text
env.object_keypoint_visible.cam_high
and
env.object_keypoint_in_frame.cam_high
```

Metrics also report normalized RMSE/MAE and pixel MAE using the dataset image
size from the config.

## Outputs

UV-only outputs are named with `uv_` prefixes to avoid mixing with the earlier
multi-target run:

```text
uv_probe_metrics.json
uv_per_source_layer_metrics.csv
uv_probe_training_curves.csv
uv_summary.csv
uv_summary.json
uv_probe_ckpts/*.pt
```

The earlier multi-target files are not produced by `run_uv_experiment.sh`.
