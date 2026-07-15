#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/data/repos/world_model_probe/configs/cosmos_probe.yaml}"
shift || true

source /data/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

export PYTHONPATH="/data/repos/world_model_probe:/data/repos/starVLA:${PYTHONPATH:-}"
cd /data/repos/world_model_probe

python -m world_model_probe.train_probe \
  --config "${CONFIG}" \
  "$@"
