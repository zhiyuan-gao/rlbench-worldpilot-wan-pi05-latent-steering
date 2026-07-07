#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  elif [[ -x "${OPENPI_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${OPENPI_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

"${PYTHON_BIN}" -m rlbench_worldpilot_wan_pi05.smoke_openvla_oft_shapes \
  --batch-size "${OPENVLA_OFT_SMOKE_BATCH_SIZE:-2}" \
  --chunk-len "${OPENVLA_OFT_NUM_ACTIONS_CHUNK}" \
  --action-dim "${OPENVLA_OFT_ACTION_DIM}" \
  --hidden-dim "${OPENVLA_OFT_SMOKE_HIDDEN_DIM:-512}" \
  --views 3 \
  --channels 16 \
  --latent-steps 6 \
  --height 32 \
  --width 32 \
  --num-heads "${WAN_FUSER_NUM_HEADS:-8}" \
  "$@"
