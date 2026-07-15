# DOMINO Cosmos Denoise Probe

Offline probing repo for checking whether frozen Cosmos-Predict2 denoising hidden
states make DOMINO future dynamics more readable than the raw no-denoise hidden
state.

Run:

```bash
bash scripts/run_experiment.sh configs/default.yaml
```

UV-only variant:

```bash
bash scripts/run_uv_experiment.sh configs/uv_only.yaml
```

The script writes all outputs under:

```text
/data/repos/domino_cosmos_denoise_probe/results/<run_id>/
```

See [EXPERIMENT.md](EXPERIMENT.md) for the original multi-target experiment
definition, and [UV_ONLY_EXPERIMENT.md](UV_ONLY_EXPERIMENT.md) for the renamed
UV-only probe path.
