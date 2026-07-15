# Reproducibility

This repository is an archival consolidation of experiments originally run in
separate working directories. It contains the code snapshot, saved configs, and
lightweight metrics, but not the external models or datasets.

## External prerequisites

You need local access to:

1. NVIDIA Cosmos-Predict2-2B-Video2World.
2. The DOM/DOMINO dataset used by the probe configs.
3. StarVLA if using the native Cosmos backbone adapter.
4. A CUDA environment compatible with the selected Cosmos release.

The original causal experiments used an environment with PyTorch 2.6.0+cu124,
Diffusers 0.37.1, and Transformers 4.57.0. These versions are historical facts,
not a fully locked environment specification.

## Track 1: frozen world-model feature probe

```bash
cd experiments/world_model_probe
bash scripts/cache_then_train.sh configs/cosmos_probe.yaml
```

Update dataset, model, cache, and checkpoint paths in the YAML file first. The
current config snapshot reflects the last current-object-position experiment;
archived result directories describe earlier delta-prediction runs.

## Track 2: causal denoise probe

```bash
cd experiments/causal_denoise_probe
bash run_eval.sh
```

The scripts retain source-machine paths. Change the Cosmos model directory,
input video, prompt, Python interpreter, and output root before running.

The two important evaluations are:

- A0: denoise a noised ground-truth future latent. This is a sanity check and is
  not causal prediction.
- A1: initialize future latent slots from pure noise and condition only on
  historical frames. This is the causal test.

## Track 3: DOMINO denoise hidden-state probe

```bash
cd experiments/domino_denoise_probe
bash scripts/run_uv_experiment.sh configs/uv_only.yaml
```

The current extractor implements native scheduler capture steps. The old
tau-based result snapshot predates this source version and cannot be reproduced
exactly from the current code.

## Result integrity

Do not compare every saved run as if only one variable changed. Across iterations,
the tasks, horizons, layers, denoising source definition, and probe target layout
also changed. Treat the repository as an experiment record and ablation archive,
not as a single controlled benchmark.

Before using the UV numbers in a paper, fix the evaluator's cross-batch weighting
to accumulate loss and errors by valid UV points, then rerun or recompute all
native-step results.
