#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/data/repos/world_model_probe/configs/cosmos_probe.yaml}"
SPLIT="${2:-all}"
shift || true
shift || true

source /data/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

export PYTHONPATH="/data/repos/world_model_probe:/data/repos/starVLA:/data/repos/dynamic-vla:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
cd /data/repos/world_model_probe

python -m world_model_probe.cache_latents \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  "$@"
