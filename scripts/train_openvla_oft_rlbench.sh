#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

EXP_NAME="${EXP_NAME:-rlbench_openvla_oft_waypoint_no_wan}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SPLIT="${SPLIT:-train}"
SAMPLE_INDEX_PATH="${SAMPLE_INDEX_PATH:-${WAN_LATENT_CACHE_ROOT}/sample_index_${SPLIT}.jsonl}"
mkdir -p "${OPENVLA_OFT_CACHE_DIR}" "${OPENVLA_OFT_RUN_ROOT}" "$(dirname "${OPENVLA_OFT_STATS_PATH}")"
export HF_HOME="${OPENVLA_OFT_HF_HOME:-${OPENVLA_OFT_CACHE_DIR}}"
export TRANSFORMERS_CACHE="${OPENVLA_OFT_TRANSFORMERS_CACHE:-${OPENVLA_OFT_CACHE_DIR}/transformers}"

if [[ -n "${OPENVLA_OFT_PYTHON:-}" ]]; then
  PYTHON_BIN="${OPENVLA_OFT_PYTHON}"
elif [[ -x "${OPENVLA_OFT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${OPENVLA_OFT_DIR}/.venv/bin/python"
elif [[ -x "${OPENPI_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${OPENPI_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

COMMON_ARGS=(
  --openvla-oft-dir "${OPENVLA_OFT_DIR}"
  --vla-path "${OPENVLA_OFT_VLA_PATH}"
  --exp-name "${EXP_NAME}"
  --run-root "${OPENVLA_OFT_RUN_ROOT}"
  --manifest-path "${MANIFEST_PATH}"
  --event-manifest-path "${EVENT_MANIFEST_PATH}"
  --sample-index-path "${SAMPLE_INDEX_PATH}"
  --goal-mode "${WAN_LATENT_GOAL_MODE}"
  --split "${SPLIT}"
  --rgb-root-200 "${RGB_ROOT_200}"
  --rgb-root-400 "${RGB_ROOT_400}"
  --lowdim-root-200 "${LOWDIM_ROOT_200}"
  --lowdim-root-400 "${LOWDIM_ROOT_400}"
  --stats-path "${OPENVLA_OFT_STATS_PATH}"
  --num-images-in-input "${OPENVLA_OFT_NUM_IMAGES_IN_INPUT}"
  --num-actions-chunk "${OPENVLA_OFT_NUM_ACTIONS_CHUNK}"
  --action-dim "${OPENVLA_OFT_ACTION_DIM}"
  --proprio-dim "${OPENVLA_OFT_PROPRIO_DIM}"
  --wan-latent-cache-root "${WAN_LATENT_CACHE_ROOT}"
  --expected-wan-num-inference-steps "${WAN_NUM_INFERENCE_STEPS}"
  --expected-wan-backend "${WAN_EXPECTED_BACKEND:-wan-diffusers}"
  --expected-wan-latent-shape "${WAN_LATENT_SHAPE}"
)

cd "${REPO_ROOT}"
if [[ " $* " == *" --dry-run "* || "${NPROC_PER_NODE}" -le 1 ]]; then
  "${PYTHON_BIN}" -m rlbench_worldpilot_wan_pi05.train_openvla_oft_rlbench \
    "${COMMON_ARGS[@]}" \
    "$@"
else
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    -m rlbench_worldpilot_wan_pi05.train_openvla_oft_rlbench \
    "${COMMON_ARGS[@]}" \
    "$@"
fi
