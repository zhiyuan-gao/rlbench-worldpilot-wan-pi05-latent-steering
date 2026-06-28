#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

: "${OPENPI_DIR:?Set OPENPI_DIR or source scripts/setup_env.sh}"

EXP_NAME="${EXP_NAME:-selected10_worldpilot_wan_pi05_torch}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
WAN_LATENT_TIME_MODE="${WAN_LATENT_TIME_MODE:-all}"
CONFIG_NAME="${CONFIG_NAME:-pi05_rlbench_waypoint_h1}"
SPLIT="${SPLIT:-train}"
SAMPLE_INDEX_PATH="${SAMPLE_INDEX_PATH:-${WAN_LATENT_CACHE_ROOT}/sample_index_${SPLIT}.jsonl}"

DRY_RUN=0
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done

if [[ "${DRY_RUN}" == "0" && -z "${PYTORCH_WEIGHT_PATH:-}" ]]; then
  cat >&2 <<'EOF'
PYTORCH_WEIGHT_PATH is required for non-dry-run PyTorch fine-tuning.

Convert a JAX pi0.5 checkpoint first if needed:
  cd /raid/home/than/zhiyuan/corl2026/pi05_baseline/openpi
  uv run examples/convert_jax_model_to_pytorch.py \
    --config-name pi05_rlbench_waypoint_h1 \
    --checkpoint-dir /path/to/jax/pi05_base_or_checkpoint \
    --output-path /path/to/pytorch/pi05_base
EOF
  exit 2
fi

COMMON_ARGS=(
  --exp-name "${EXP_NAME}"
  --lerobot-repo-id "${LEROBOT_REPO_ID}"
  --manifest-path "${MANIFEST_PATH}"
  --event-manifest-path "${EVENT_MANIFEST_PATH}"
  --goal-mode "${WAN_LATENT_GOAL_MODE}"
  --sample-index-path "${SAMPLE_INDEX_PATH}"
  --wan-latent-cache-root "${WAN_LATENT_CACHE_ROOT}"
  --split "${SPLIT}"
  --time-mode "${WAN_LATENT_TIME_MODE}"
  --expected-wan-num-inference-steps "${WAN_NUM_INFERENCE_STEPS}"
)
if [[ -n "${PYTORCH_WEIGHT_PATH:-}" ]]; then
  COMMON_ARGS+=(--pytorch-weight-path "${PYTORCH_WEIGHT_PATH}")
fi

cd "${OPENPI_DIR}"
if [[ "${DRY_RUN}" == "1" || "${NPROC_PER_NODE}" -le 1 ]]; then
  uv run python -m rlbench_worldpilot_wan_pi05.train_torch \
    "${CONFIG_NAME}" \
    "${COMMON_ARGS[@]}" \
    "$@"
else
  uv run torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    -m rlbench_worldpilot_wan_pi05.train_torch \
    "${CONFIG_NAME}" \
    "${COMMON_ARGS[@]}" \
    "$@"
fi
