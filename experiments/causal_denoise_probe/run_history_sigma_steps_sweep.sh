#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/starVLA/bin/python}"
MODEL_DIR="${MODEL_DIR:-/data/repos/starVLA/playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World}"
VIDEO_PATH="${VIDEO_PATH:-/data/repos/cosmos-predict2.5/assets/base/robot_pouring.mp4}"
PROMPT_PATH="${PROMPT_PATH:-/data/repos/cosmos-predict2.5/assets/base/robot_pouring.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/history_sigma_steps_sweep}"

# Pixel-frame condition counts. Because Cosmos VAE downsamples time by 4,
# these are the meaningful boundaries for adding one more condition latent slot:
# 1 -> 1 latent slot, 5 -> 2, 9 -> 3, 13 -> 4.
COND_FRAMES_LIST="${COND_FRAMES_LIST:-1,5,9,13}"
FUTURE_PIXEL_FRAMES="${FUTURE_PIXEL_FRAMES:-4}"

# Sweep grid. Keep this moderate at full 1280x704; it is multiplicative with history counts.
SIGMAS="${SIGMAS:-0.002,0.05,0.1,0.2,0.3,0.5,0.8,1.0,2.0,5.0}"
DENOISE_STEPS="${DENOISE_STEPS:-1,3,5,10,20,35}"

FRAME_START="${FRAME_START:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-704}"
FPS="${FPS:-16}"
SEED="${SEED:-123}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
HOOK_LAYER="${HOOK_LAYER:--1}"
VAE_SAMPLE_MODE="${VAE_SAMPLE_MODE:-sample}"
SAVE_LATENTS="${SAVE_LATENTS:-0}"
SAVE_LATENT_DTYPE="${SAVE_LATENT_DTYPE:-bfloat16}"

EXTRA_ARGS=()
if [[ "${SAVE_LATENTS}" == "1" || "${SAVE_LATENTS}" == "true" || "${SAVE_LATENTS}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--save-latents --save-latent-dtype "${SAVE_LATENT_DTYPE}")
fi

mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a COND_ARRAY <<< "${COND_FRAMES_LIST}"
for cond_frames_raw in "${COND_ARRAY[@]}"; do
  cond_frames="$(echo "${cond_frames_raw}" | xargs)"
  if [[ -z "${cond_frames}" ]]; then
    continue
  fi

  output_dir="${OUTPUT_ROOT}/cond_${cond_frames}_future_${FUTURE_PIXEL_FRAMES}"
  echo "[RUN] cond_frames=${cond_frames} future_pixel_frames=${FUTURE_PIXEL_FRAMES} output=${output_dir}"

  PYTHON_BIN="${PYTHON_BIN}" \
  MODEL_DIR="${MODEL_DIR}" \
  VIDEO_PATH="${VIDEO_PATH}" \
  PROMPT_PATH="${PROMPT_PATH}" \
  OUTPUT_DIR="${output_dir}" \
  "${SCRIPT_DIR}/run_eval.sh" \
    --cond-frames "${cond_frames}" \
    --future-pixel-frames "${FUTURE_PIXEL_FRAMES}" \
    --sigmas "${SIGMAS}" \
    --denoise-steps "${DENOISE_STEPS}" \
    --frame-start "${FRAME_START}" \
    --frame-stride "${FRAME_STRIDE}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --fps "${FPS}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --hook-layer "${HOOK_LAYER}" \
    --vae-sample-mode "${VAE_SAMPLE_MODE}" \
    "${EXTRA_ARGS[@]}"
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/aggregate_history_sweep.py" --output-root "${OUTPUT_ROOT}"
