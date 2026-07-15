#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-/data/repos/starVLA/playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World}"
VIDEO_PATH="${VIDEO_PATH:-/data/repos/cosmos-predict2.5/assets/base/robot_pouring.mp4}"
PROMPT_PATH="${PROMPT_PATH:-/data/repos/cosmos-predict2.5/assets/base/robot_pouring.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/robot_pouring}"

# The default /data/miniconda3 Python on this machine does not include torch/diffusers.
# Use PYTHON_BIN to point at the project environment. This PYTHONPATH fallback only helps
# if torch/transformers are already installed and diffusers is missing.
LOCAL_DIFFUSERS_CACHE="/data/uv-cache/archive-v0/MCt77ZsjPTHRGqy1"
if [[ -d "${LOCAL_DIFFUSERS_CACHE}" ]]; then
  export PYTHONPATH="${LOCAL_DIFFUSERS_CACHE}:${PYTHONPATH:-}"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/cosmos_causal_probe.py" \
  --model-dir "${MODEL_DIR}" \
  --video "${VIDEO_PATH}" \
  --prompt "${PROMPT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
