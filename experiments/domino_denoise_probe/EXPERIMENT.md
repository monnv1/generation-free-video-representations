# Experiment: Latent-Space Cosmos Denoising Probe

## Goal

This experiment asks one narrow question:

```text
Does frozen Cosmos-Predict2 denoising make DOMINO future dynamics more readable
from the model's hidden states?
```

It does not train an action head, does not fine-tune Cosmos, and does not run
closed-loop control.

## Data

Default tasks:

```text
grab_roller
move_playingcard_away
click_bell
```

Only the head camera is used:

```text
observation.images.cam_high
```

The input history is 5 frames:

```text
o_{t-4:t}
```

Future pixels are never fed to Cosmos. Future labels are only used as probe
supervision.

## Representations Compared

All probes read Cosmos hidden-space vectors. There is no pixel-space baseline.

### 1. Raw No-Denoise Baseline

```text
history frames -> VAE clean history latent -> Cosmos transformer at timestep 0
```

The resulting hidden vector is named:

```text
raw_no_denoise
```

It is used for two controls:

```text
raw_no_denoise -> current labels
raw_no_denoise -> future labels
```

The first must work; otherwise the extraction or probe is broken. The second is
the control for how much future is already readable without denoising.

### 2. Denoising Hidden States

For denoising, the model input contains:

```text
clean history latent slots + random future latent slots
```

The condition mask marks the history latent slots as observed. Future slots are
random noise; no ground-truth future frames are encoded.

At each tau:

```text
tau in {0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.1, 0.0}
```

the code captures conditional-branch transformer hidden states from layers:

```text
6, 14, 27
```

These are early, middle, and late layers of the 28-layer Cosmos transformer.

CFG is enabled by default:

```text
cfg_scale = 7.0
negative_prompt = ""
fps = 16
```

The unconditional branch is used to update the denoising sample, but the saved
probe representation is the conditional hidden state.

## Probe Targets

Current labels:

```text
object_keypoint_xyz[t]
object_keypoint_uv[t]
object_keypoint_depth[t]
gripper_contact[t]
task_success[t]
```

Future labels at horizons:

```text
t+1, t+2, t+4, t+8, t+15
```

Targets:

```text
future object 3D keypoint
future object 2D keypoint
future depth
future object velocity
future contact flags
future success flags
time-to-contact bucket
```

2D losses are masked by DOMINO's visibility and in-frame flags. This matters for
click tasks, where keypoint visibility is often low.

## Expected Readout

The important comparison is:

```text
denoise_tau=<tau>, layer=<l> -> future labels
vs
raw_no_denoise, layer=<l> -> future labels
```

If denoised hidden states improve future prediction over raw hidden states, the
result supports the claim that the denoising process makes future dynamics more
readable in Cosmos latent space.

If raw hidden states already predict future labels well, the task may be
predictable from current state alone, reducing the interpretability of denoising
gains.

If raw hidden states cannot predict current labels, the experiment is invalid
until feature extraction, pooling, or layer hooks are fixed.

## Outputs

Each run writes:

```text
config.yaml
slice_index.parquet
labels.npz
features.npy
feature_meta.json
probe_metrics.json
probe_training_curves.csv
per_source_layer_metrics.csv
summary.csv
summary.json
probe_ckpts/*.pt
```

`summary.csv` contains denoised-vs-raw improvements per layer and tau. A positive
`loss_improvement_vs_raw` means the denoising hidden state outperformed the raw
no-denoise hidden baseline for future dynamics.


## Probe Checkpoints And Curves

By default, every probe saves the best checkpoint by validation loss:

```text
probe_ckpts/<target>__<source>__layer_<l>.pt
```

Each checkpoint contains:

```text
model_state_dict
target/source/layer metadata
feature normalization mean/std
best validation metrics
test metrics
```

Optimizer state is not saved by default to keep storage small. Turn on:

```yaml
probe:
  save_optimizer: true
```

only if the probe needs to be resumed exactly.

Training curves are saved in:

```text
probe_training_curves.csv
```

Each row is one epoch for one probe and includes `train_loss`, `val_loss`, and
the validation readout metrics. Use this file to check whether a probe converged,
hit early stopping too soon, or was still improving.
