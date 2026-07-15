#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/data/repos/world_model_probe/configs/cosmos_probe.yaml}"
CACHE_SPLIT="${2:-all}"
shift || true
shift || true

CACHE_ARGS=()
TRAIN_ARGS=()
TARGET="cache"
for arg in "$@"; do
  if [[ "${arg}" == "--" ]]; then
    TARGET="train"
    continue
  fi
  if [[ "${TARGET}" == "cache" ]]; then
    CACHE_ARGS+=("${arg}")
  else
    TRAIN_ARGS+=("${arg}")
  fi
done

/data/repos/world_model_probe/scripts/cache_latents.sh \
  "${CONFIG}" \
  "${CACHE_SPLIT}" \
  "${CACHE_ARGS[@]}"

/data/repos/world_model_probe/scripts/train_probe.sh \
  "${CONFIG}" \
  "${TRAIN_ARGS[@]}"
