#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

: "${OPENPI_DIR:?Set OPENPI_DIR or source scripts/hpc_paths.sh}"
: "${MANIFEST_PATH:?Set MANIFEST_PATH or source scripts/hpc_paths.sh}"
: "${EVENT_MANIFEST_PATH:?Set EVENT_MANIFEST_PATH or source scripts/hpc_paths.sh}"
: "${WAN_LATENT_CACHE_ROOT:?Set WAN_LATENT_CACHE_ROOT or source scripts/hpc_paths.sh}"
: "${PI05_FT_PYTORCH_WEIGHT_PATH:?Set this to the converted RLBench-finetuned pi0.5 checkpoint, not pi0.5 base}"

CONFIG_NAME="${CONFIG_NAME:-pi05_rlbench_waypoint_h1}"
EXP_NAME="${EXP_NAME:-selected10_fcrf_v1_pilot2k}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SPLIT="${SPLIT:-train}"
SAMPLE_INDEX_PATH="${SAMPLE_INDEX_PATH:-${WAN_LATENT_CACHE_ROOT}/sample_index_${SPLIT}.jsonl}"
WAN_EXPECTED_BACKEND="${WAN_EXPECTED_BACKEND:-wan-diffusers}"

COMMON_ARGS=(
  --exp-name "${EXP_NAME}"
  --lerobot-repo-id "${LEROBOT_REPO_ID}"
  --manifest-path "${MANIFEST_PATH}"
  --event-manifest-path "${EVENT_MANIFEST_PATH}"
  --goal-mode "${WAN_LATENT_GOAL_MODE:-event_end}"
  --sample-index-path "${SAMPLE_INDEX_PATH}"
  --wan-latent-cache-root "${WAN_LATENT_CACHE_ROOT}"
  --split "${SPLIT}"
  --pi05-ft-weight-path "${PI05_FT_PYTORCH_WEIGHT_PATH}"
  --expected-wan-num-inference-steps "${WAN_NUM_INFERENCE_STEPS:-1}"
  --expected-wan-backend "${WAN_EXPECTED_BACKEND}"
  --expected-wan-latent-shape "${WAN_LATENT_SHAPE:-3,16,6,32,32}"
  --batch-size 128
  --num-train-steps 2000
  --save-interval 500
  --keep-period 500
  --lr-schedule.warmup-steps 200
  --residual-penalty 1e-4
)

cd "${OPENPI_DIR}"
if [[ "${NPROC_PER_NODE}" -le 1 ]]; then
  uv run python -m rlbench_worldpilot_wan_pi05.train_fcrf_v1 \
    "${CONFIG_NAME}" \
    "${COMMON_ARGS[@]}" \
    "$@"
else
  uv run torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    -m rlbench_worldpilot_wan_pi05.train_fcrf_v1 \
    "${CONFIG_NAME}" \
    "${COMMON_ARGS[@]}" \
    "$@"
fi
