#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

: "${OPENPI_DIR:?Set OPENPI_DIR or source scripts/setup_env.sh}"

EXP_NAME="${EXP_NAME:-selected10_worldpilot_wan_pi05_torch}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CONFIG_NAME="${CONFIG_NAME:-pi05_rlbench_waypoint_h1}"
SPLIT="${SPLIT:-train}"
SAMPLE_INDEX_PATH="${SAMPLE_INDEX_PATH:-${WAN_LATENT_CACHE_ROOT}/sample_index_${SPLIT}.jsonl}"
WAN_EXPECTED_BACKEND="${WAN_EXPECTED_BACKEND:-wan-diffusers}"

COMMON_ARGS=(
  "${CONFIG_NAME}"
  --exp-name "${EXP_NAME}"
  --lerobot-repo-id "${LEROBOT_REPO_ID}"
  --manifest-path "${MANIFEST_PATH}"
  --event-manifest-path "${EVENT_MANIFEST_PATH}"
  --goal-mode "${WAN_LATENT_GOAL_MODE}"
  --sample-index-path "${SAMPLE_INDEX_PATH}"
  --wan-latent-cache-root "${WAN_LATENT_CACHE_ROOT}"
  --split "${SPLIT}"
  --expected-wan-num-inference-steps "${WAN_NUM_INFERENCE_STEPS}"
  --expected-wan-backend "${WAN_EXPECTED_BACKEND}"
  --eval-only
)
if [[ -n "${PYTORCH_WEIGHT_PATH:-}" ]]; then
  COMMON_ARGS+=(--pytorch-weight-path "${PYTORCH_WEIGHT_PATH}")
fi

cd "${OPENPI_DIR}"
if [[ "${NPROC_PER_NODE}" -le 1 ]]; then
  uv run python -m rlbench_worldpilot_wan_pi05.train_torch \
    "${COMMON_ARGS[@]}" \
    "$@"
else
  uv run torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    -m rlbench_worldpilot_wan_pi05.train_torch \
    "${COMMON_ARGS[@]}" \
    "$@"
fi
