#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/uv_only.yaml}"
RUN_ID="${RUN_ID:-domino_cosmos_uv_probe_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

pids=()
cleanup_workers() {
  if [[ "${#pids[@]}" -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup_workers INT TERM HUP EXIT

echo "[1/4] Building DOMINO slice/label cache"
python -m domino_cosmos_probe.build_slices --config "${CONFIG_PATH}" --run-id "${RUN_ID}"

EXTRACT_WORKERS="${EXTRACT_WORKERS:-$(python - <<'PY' "${CONFIG_PATH}"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(int((cfg.get("extract") or {}).get("workers", 1)))
PY
)}"

if [[ "${EXTRACT_WORKERS}" -le 1 ]]; then
  echo "[2/4] Extracting frozen Cosmos raw/denoise features"
  python -m domino_cosmos_probe.extract_features --config "${CONFIG_PATH}" --run-id "${RUN_ID}"
else
  echo "[2/4] Extracting frozen Cosmos raw/denoise features with ${EXTRACT_WORKERS} workers"
  python -m domino_cosmos_probe.extract_features \
    --config "${CONFIG_PATH}" \
    --run-id "${RUN_ID}" \
    --num-shards "${EXTRACT_WORKERS}" \
    --init-only

  pids=()
  for shard_id in $(seq 0 $((EXTRACT_WORKERS - 1))); do
    python -m domino_cosmos_probe.extract_features \
      --config "${CONFIG_PATH}" \
      --run-id "${RUN_ID}" \
      --num-shards "${EXTRACT_WORKERS}" \
      --shard-id "${shard_id}" &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  pids=()
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
fi

echo "[3/4] Training UV-only current/future probes"
python -m domino_cosmos_probe.train_uv_probes --config "${CONFIG_PATH}" --run-id "${RUN_ID}"

RUN_DIR="$(python - <<'PY' "${CONFIG_PATH}" "${RUN_ID}"
from pathlib import Path
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(Path(cfg["run"]["output_root"]) / sys.argv[2])
PY
)"

echo "[4/4] Summarizing UV-only results"
python -m domino_cosmos_probe.summarize_uv --run-dir "${RUN_DIR}"

echo "Done. UV-only results: ${RUN_DIR}"
