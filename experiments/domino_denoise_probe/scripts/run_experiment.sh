#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/default.yaml}"
RUN_ID="${RUN_ID:-domino_cosmos_denoise_probe_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

echo "[1/4] Building DOMINO slice/label cache"
python -m domino_cosmos_probe.build_slices --config "${CONFIG_PATH}" --run-id "${RUN_ID}"

echo "[2/4] Extracting frozen Cosmos raw/denoise features"
python -m domino_cosmos_probe.extract_features --config "${CONFIG_PATH}" --run-id "${RUN_ID}"

echo "[3/4] Training current/future probes"
python -m domino_cosmos_probe.train_probes --config "${CONFIG_PATH}" --run-id "${RUN_ID}"

RUN_DIR="$(python - <<'PY' "${CONFIG_PATH}" "${RUN_ID}"
from pathlib import Path
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(Path(cfg["run"]["output_root"]) / sys.argv[2])
PY
)"

echo "[4/4] Summarizing results"
python -m domino_cosmos_probe.summarize --run-dir "${RUN_DIR}"

echo "Done. Results: ${RUN_DIR}"
