#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/starVLA/bin/python}"
MODEL_DIR="${MODEL_DIR:-/data/repos/starVLA/playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World}"
VIDEO_PATH="${VIDEO_PATH:-/data/repos/cosmos-predict2.5/assets/base/robot_pouring.mp4}"
PROMPT_PATH="${PROMPT_PATH:-/data/repos/cosmos-predict2.5/assets/base/robot_pouring.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/cfg_sequence_sweep}"

# Full-resolution 1280x704 can become expensive quickly because token count grows
# with (condition latent slots + future latent slots). Start conservative on 48GB.
COND_FRAMES_LIST="${COND_FRAMES_LIST:-1,5,9}"
FUTURE_LATENT_SLOTS_LIST="${FUTURE_LATENT_SLOTS_LIST:-1,2,4}"
NEGATIVE_PROMPT_MODES="${NEGATIVE_PROMPT_MODES:-empty,quality}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-1,3,5,7}"
SIGMAS="${SIGMAS:-0.5,0.8,1.0,2.0}"
DENOISE_STEPS="${DENOISE_STEPS:-1,3,5,10,20}"

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
QUALITY_NEGATIVE_PROMPT_PATH="${QUALITY_NEGATIVE_PROMPT_PATH:-${SCRIPT_DIR}/negative_prompt_quality.txt}"

mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a COND_ARRAY <<< "${COND_FRAMES_LIST}"
IFS=',' read -r -a SLOT_ARRAY <<< "${FUTURE_LATENT_SLOTS_LIST}"
IFS=',' read -r -a NEG_ARRAY <<< "${NEGATIVE_PROMPT_MODES}"

for cond_frames_raw in "${COND_ARRAY[@]}"; do
  cond_frames="$(echo "${cond_frames_raw}" | xargs)"
  [[ -z "${cond_frames}" ]] && continue

  for slots_raw in "${SLOT_ARRAY[@]}"; do
    future_latent_slots="$(echo "${slots_raw}" | xargs)"
    [[ -z "${future_latent_slots}" ]] && continue
    future_pixel_frames=$(( future_latent_slots * 4 ))

    for neg_raw in "${NEG_ARRAY[@]}"; do
      neg_mode="$(echo "${neg_raw}" | xargs)"
      [[ -z "${neg_mode}" ]] && continue

      case "${neg_mode}" in
        empty)
          negative_prompt_arg=""
          ;;
        quality)
          negative_prompt_arg="${QUALITY_NEGATIVE_PROMPT_PATH}"
          ;;
        *)
          if [[ -f "${neg_mode}" ]]; then
            negative_prompt_arg="${neg_mode}"
          else
            echo "Unknown NEGATIVE_PROMPT mode/path: ${neg_mode}" >&2
            exit 2
          fi
          ;;
      esac

      output_dir="${OUTPUT_ROOT}/cond_${cond_frames}_slots_${future_latent_slots}_neg_${neg_mode}"
      echo "[RUN] cond_frames=${cond_frames} future_latent_slots=${future_latent_slots} neg=${neg_mode} cfg=${GUIDANCE_SCALES} output=${output_dir}"

      PYTHON_BIN="${PYTHON_BIN}" \
      MODEL_DIR="${MODEL_DIR}" \
      VIDEO_PATH="${VIDEO_PATH}" \
      PROMPT_PATH="${PROMPT_PATH}" \
      OUTPUT_DIR="${output_dir}" \
      "${SCRIPT_DIR}/run_eval.sh" \
        --cond-frames "${cond_frames}" \
        --future-pixel-frames "${future_pixel_frames}" \
        --future-latent-slots "${future_latent_slots}" \
        --negative-prompt "${negative_prompt_arg}" \
        --guidance-scales "${GUIDANCE_SCALES}" \
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
  done
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/aggregate_history_sweep.py" --output-root "${OUTPUT_ROOT}"
