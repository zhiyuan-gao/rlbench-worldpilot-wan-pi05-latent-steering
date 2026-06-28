#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

: "${OPENPI_DIR:?Set OPENPI_DIR or source scripts/setup_env.sh}"

CONFIG_NAME="${CONFIG_NAME:-pi05_rlbench_waypoint_h1}"
EXP_NAME="${EXP_NAME:-selected10_worldpilot_wan_pi05_torch}"
SPLIT="${SPLIT:-val}"
WAN_LATENT_BACKEND="${WAN_LATENT_BACKEND:-wan-diffusers}"
WAN_LATENT_TIME_MODE="${WAN_LATENT_TIME_MODE:-all}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-./checkpoints}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-./assets}"
ONLINE_EVAL_OUT="${ONLINE_EVAL_OUT:-${REPO_ROOT}/online_eval/${EXP_NAME}_${SPLIT}.jsonl}"

EXTRA_ARGS=()
if [[ -n "${EVAL_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--eval-checkpoint "${EVAL_CHECKPOINT}")
fi
if [[ -n "${WAN_LORA_DIR:-}" ]]; then
  EXTRA_ARGS+=(--wan-lora-dir "${WAN_LORA_DIR}")
fi

case ":${PYTHONPATH:-}:" in
  *":${RLBENCH_ROOT}:"*) ;;
  *) export PYTHONPATH="${RLBENCH_ROOT}:${PYTHONPATH:-}" ;;
esac

if [[ -n "${COPPELIASIM_ROOT:-}" ]]; then
  export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${COPPELIASIM_ROOT}/lib:${LD_LIBRARY_PATH:-}"
  export PATH="${COPPELIASIM_ROOT}:${PATH}"
fi

cd "${OPENPI_DIR}"
uv run python -m rlbench_worldpilot_wan_pi05.eval_online_rlbench \
  "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --manifest-path "${MANIFEST_PATH}" \
  --event-manifest-path "${EVENT_MANIFEST_PATH}" \
  --split "${SPLIT}" \
  --out "${ONLINE_EVAL_OUT}" \
  --rlbench-root "${RLBENCH_ROOT}" \
  --lowdim-root-200 "${LOWDIM_ROOT_200}" \
  --lowdim-root-400 "${LOWDIM_ROOT_400}" \
  --rgb-root-200 "${RGB_ROOT_200}" \
  --rgb-root-400 "${RGB_ROOT_400}" \
  --wan-backend "${WAN_LATENT_BACKEND}" \
  --wan-base-model "${WAN_BASE_MODEL}" \
  --wan-output-layout "${WAN_OUTPUT_LAYOUT:-bcthw}" \
  --wan-num-inference-steps "${WAN_NUM_INFERENCE_STEPS}" \
  --wan-latent-time-mode "${WAN_LATENT_TIME_MODE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
