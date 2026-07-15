# Cosmos Implicit World Model for Embodied Multimodal Reasoning

Research code and lightweight experiment artifacts for reusing a pretrained video
world model as an implicit dynamics backbone, without explicitly generating future
video at policy inference time.

## Motivation

Video world models learn useful spatial, temporal, and language-conditioned
knowledge, but autoregressive or diffusion-based future video generation is too
expensive for many real-time embodied-control settings. This project studies a
shorter path:

```text
history RGB + language instruction + robot state
                    |
        frozen video world-model backbone
                    |
             latent / hidden tokens
                    |
        dynamics readout or policy adapter
                    |
      object motion, contact, or robot action
```

The central question is whether the pretrained model's internal tokens can expose
useful future dynamics directly, avoiding a full multi-step pixel-generation
chain.

## What Is Included

This repository consolidates three experiment tracks:

| Track | Question | Code | Lightweight results |
|---|---|---|---|
| Frozen feature probe | Can raw Cosmos tokens predict robot and object dynamics? | [`experiments/world_model_probe`](experiments/world_model_probe) | [`results/world_model_probe`](results/world_model_probe) |
| Causal denoise probe | Can short-chain future denoising beat a persistence latent? | [`experiments/causal_denoise_probe`](experiments/causal_denoise_probe) | [`results/causal_denoise_probe`](results/causal_denoise_probe) |
| DOMINO denoise probe | Are denoising hidden states more readable than raw hidden states? | [`experiments/domino_denoise_probe`](experiments/domino_denoise_probe) | [`results/domino_denoise_probe`](results/domino_denoise_probe) |

The repository intentionally excludes model weights, datasets, latent caches,
probe checkpoints, and multi-gigabyte feature arrays.

## Main Findings

### 1. Raw world-model tokens contain limited dynamics information

On the frozen-feature probe, robot-arm motion was substantially more readable
than object motion:

| Target | Probe L2 | Persistence L2 | Improvement |
|---|---:|---:|---:|
| Object position | 0.0361 | 0.0363 | 0.72% |
| Object velocity | 0.1030 | 0.1043 | 1.17% |
| Arm position | 0.0606 | 0.0830 | 27.0% |

Object-motion probes frequently collapsed toward near-zero predictions on
large-motion samples. See
[`object_motion_diagnostics_eval.md`](results/world_model_probe/object_motion_diagnostics/object_motion_diagnostics_eval.md).

### 2. Naive short-chain denoising did not beat persistence

The causal probe initializes unknown future latent slots from pure noise and
compares the denoised latent with a baseline that repeats the final condition
latent. Across 240 history/sigma/step combinations, no run beat persistence.

| Condition frames | Best predicted cosine | Persistence cosine | Gain |
|---:|---:|---:|---:|
| 1 | 0.2326 | 0.8280 | -0.5954 |
| 5 | 0.1943 | 0.8893 | -0.6950 |
| 9 | 0.1103 | 0.9386 | -0.8283 |
| 13 | 0.0722 | 0.9647 | -0.8925 |

CFG and a quality negative prompt improved the best gain to `-0.4099`, but did
not reverse the result. Multi-slot denoising was also worse than persistence.

### 3. Native denoising states were less readable than raw states

The initial short-horizon UV-only experiment found a local improvement at one
layer and noise level. After switching to the native 35-step schedule,
per-future-slot alignment, five transformer layers, and 0.5-3 second horizons,
all 225 denoising layer/horizon combinations lost to the raw representation.

| Horizon | Best raw UV MAE | Best denoised UV MAE |
|---:|---:|---:|
| 8 | 21.35 px | 39.60 px |
| 16 | 19.62 px | 37.68 px |
| 24 | 17.07 px | 36.45 px |
| 36 | 14.25 px | 29.88 px |
| 45 | 13.64 px | 27.38 px |

These results do not support simply truncating the original denoising chain.
They motivate distilling a control-oriented dynamics latent from the frozen
backbone instead.

## Repository Layout

```text
.
|-- experiments/
|   |-- world_model_probe/
|   |-- causal_denoise_probe/
|   `-- domino_denoise_probe/
|-- results/
|   |-- world_model_probe/
|   |-- causal_denoise_probe/
|   `-- domino_denoise_probe/
|-- docs/
|   |-- EXPERIMENT_SUMMARY_ZH.md
|   |-- CAUSAL_EXPERIMENT_NOTES_ZH.md
|   |-- REPRODUCIBILITY.md
|   `-- PROVENANCE.md
`-- assets/
```

## Reproduction

The experiments depend on external assets that are not redistributed here:

- NVIDIA Cosmos-Predict2-2B-Video2World weights
- DOM/DOMINO robot datasets
- the original StarVLA/Cosmos integration
- a CUDA-capable environment

Original experiment configs contain `/data/...` paths from the source machine.
Set the paths for your own environment before running. Start with
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Important Limitations

- The causal generation study used one `robot_pouring` video and one seed.
- The DOMINO UV evaluator has a cross-batch valid-point weighting issue. The first
  UV run includes a corrected recomputation; native runs should be recomputed
  before publication.
- `click_bell` has no valid UV labels in these saved runs, so UV conclusions are
  effectively based on two tasks.
- Best configurations were selected from exploratory test sweeps without
  confidence intervals.
- The old tau-based extractor was overwritten by the later native-step version;
  its saved metrics are retained, but it is not exactly reproducible from the
  current source snapshot.

## Documentation

- [Full experiment summary in Chinese](docs/EXPERIMENT_SUMMARY_ZH.md)
- [Original causal experiment notes](docs/CAUSAL_EXPERIMENT_NOTES_ZH.md)
- [Artifact provenance and exclusions](docs/PROVENANCE.md)
- [Figure inventory](docs/FIGURE_INDEX_ZH.md)

## License

No license has been selected for this consolidated repository. Add a license
before making the repository public. External models, datasets, and upstream
projects remain subject to their own licenses and terms.
