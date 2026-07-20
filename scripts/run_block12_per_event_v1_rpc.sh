#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

CONFIG_NAME="${CONFIG_NAME:-pi05_rlbench_waypoint_h1}"
EXP_NAME="${EXP_NAME:-block12_per_event_v1}"
EVAL_SEED="${EVAL_SEED:-0}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/block12_per_event_v1_seed${EVAL_SEED}}"

OPENPI_PY="${OPENPI_PY:-${OPENPI_DIR}/.venv/bin/python}"
RLBENCH_PY="${RLBENCH_PY:-}"
PYREP_ROOT="${PYREP_ROOT:-}"
COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$(dirname "${EVAL_CHECKPOINT:-.}")}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${CHECKPOINT_BASE_DIR}}"

WAN_HOST="${WAN_HOST:-127.0.0.1}"
WAN_PORT="${WAN_PORT:-18788}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-18787}"
WAN_VISIBLE_DEVICES="${WAN_VISIBLE_DEVICES:-0,1}"
POLICY_VISIBLE_DEVICES="${POLICY_VISIBLE_DEVICES:-1}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda:0}"
WAN_DEVICE_MAP="${WAN_DEVICE_MAP:-balanced}"
WAN_RPC_TIMEOUT_SEC="${WAN_RPC_TIMEOUT_SEC:-900}"

DISPLAY_ID="${DISPLAY_ID:-:190}"
START_XVFB="${START_XVFB:-1}"
MONITOR_GPU="${MONITOR_GPU:-1}"
POLICY_PYTHONPATH_EXTRA="${POLICY_PYTHONPATH_EXTRA:-}"
CLIENT_PYTHONPATH_EXTRA="${CLIENT_PYTHONPATH_EXTRA:-}"

DEFAULT_TASKS=(
  meat_off_grill
  open_drawer
  push_buttons
  put_money_in_safe
  reach_and_drag
  slide_block_to_target
  stack_cups
  stack_wine
  sweep_to_dustpan
  turn_tap
)
if [[ -n "${TASKS:-}" ]]; then
  read -r -a TASK_LIST <<<"${TASKS}"
else
  TASK_LIST=("${DEFAULT_TASKS[@]}")
fi

for required_name in RLBENCH_PY PYREP_ROOT COPPELIASIM_ROOT EVAL_CHECKPOINT WAN_BASE_MODEL; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "Set ${required_name} before running this script." >&2
    exit 2
  fi
done
for required_file in "${OPENPI_PY}" "${RLBENCH_PY}"; do
  if [[ ! -x "${required_file}" ]]; then
    echo "Missing executable: ${required_file}" >&2
    exit 2
  fi
done
for required_path in \
  "${EVAL_CHECKPOINT}" \
  "${WAN_BASE_MODEL}" \
  "${MANIFEST_PATH}" \
  "${EVENT_MANIFEST_PATH}" \
  "${LOWDIM_ROOT_200}" \
  "${LOWDIM_ROOT_400}" \
  "${RGB_ROOT_200}" \
  "${RGB_ROOT_400}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 2
  fi
done
if [[ -n "${WAN_LORA_DIR:-}" && ! -e "${WAN_LORA_DIR}" ]]; then
  echo "Missing WAN_LORA_DIR: ${WAN_LORA_DIR}" >&2
  exit 2
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output directory: ${OUT_DIR}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
touch "${OUT_DIR}/RUNNING"

WAN_PID=
POLICY_PID=
XVFB_PID=
GPU_MON_PID=
CLIENT_PID=

cleanup() {
  local rc=$?
  local pid
  trap - EXIT INT TERM
  for pid in "${CLIENT_PID}" "${GPU_MON_PID}" "${POLICY_PID}" "${WAN_PID}" "${XVFB_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait "${CLIENT_PID}" "${GPU_MON_PID}" "${POLICY_PID}" "${WAN_PID}" "${XVFB_PID}" 2>/dev/null || true
  rm -f "${OUT_DIR}/RUNNING"
  if [[ ${rc} -eq 0 ]]; then
    touch "${OUT_DIR}/DONE"
  else
    printf '%s\n' "${rc}" >"${OUT_DIR}/FAILED"
  fi
  exit "${rc}"
}
trap cleanup EXIT INT TERM

wait_for_health() {
  local name=$1
  local url=$2
  local pid=$3
  local attempts=${4:-180}
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited before becoming healthy" >&2
      return 1
    fi
    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "${name} healthy after ${attempt} checks"
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for ${name}: ${url}" >&2
  return 1
}

if [[ "${START_XVFB}" == "1" ]]; then
  Xvfb "${DISPLAY_ID}" -screen 0 1280x1024x24 -ac -nolisten tcp >"${OUT_DIR}/xvfb.log" 2>&1 &
  XVFB_PID=$!
fi
if [[ "${MONITOR_GPU}" == "1" ]]; then
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits \
    -l 5 >"${OUT_DIR}/gpu_usage.csv" &
  GPU_MON_PID=$!
fi

POLICY_PYTHONPATH="${REPO_ROOT}/src:${OPENPI_DIR}/src:${POLICY_PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
WAN_LORA_ARGS=()
if [[ -n "${WAN_LORA_DIR:-}" ]]; then
  WAN_LORA_ARGS+=(--wan-lora-dir "${WAN_LORA_DIR}")
fi

env \
  CUDA_VISIBLE_DEVICES="${WAN_VISIBLE_DEVICES}" \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="${POLICY_PYTHONPATH}" \
  "${OPENPI_PY}" -m rlbench_worldpilot_wan_pi05.serve_wan_latent_rpc \
    --host "${WAN_HOST}" \
    --port "${WAN_PORT}" \
    --wan-base-model "${WAN_BASE_MODEL}" \
    "${WAN_LORA_ARGS[@]}" \
    --wan-height 256 \
    --wan-view-width 256 \
    --wan-num-frames 21 \
    --wan-num-inference-steps 1 \
    --wan-output-layout bcthw \
    --wan-dtype bf16 \
    --wan-device-map "${WAN_DEVICE_MAP}" \
    >"${OUT_DIR}/wan_rpc.log" 2>&1 &
WAN_PID=$!
wait_for_health WAN "http://${WAN_HOST}:${WAN_PORT}/health" "${WAN_PID}"

env \
  CUDA_VISIBLE_DEVICES="${POLICY_VISIBLE_DEVICES}" \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="${POLICY_PYTHONPATH}" \
  "${OPENPI_PY}" -m rlbench_worldpilot_wan_pi05.serve_policy_rpc \
    "${CONFIG_NAME}" \
    --host "${POLICY_HOST}" \
    --port "${POLICY_PORT}" \
    --exp-name "${EXP_NAME}" \
    --eval-checkpoint "${EVAL_CHECKPOINT}" \
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
    --assets-base-dir "${ASSETS_BASE_DIR}" \
    --policy-device "${POLICY_DEVICE}" \
    --pytorch-training-precision bfloat16 \
    --wan-latent-shape "${WAN_LATENT_SHAPE}" \
    --wan-steering-mode block \
    --wan-steering-block 12 \
    --wan-steering-gate auto \
    --action-num-steps 10 \
    --wan-rpc-url "http://${WAN_HOST}:${WAN_PORT}/latent" \
    --wan-rpc-timeout-sec "${WAN_RPC_TIMEOUT_SEC}" \
    >"${OUT_DIR}/policy_rpc.log" 2>&1 &
POLICY_PID=$!
wait_for_health policy "http://${POLICY_HOST}:${POLICY_PORT}/health" "${POLICY_PID}"

CLIENT_PYTHONPATH="${RLBENCH_ROOT}:${PYREP_ROOT}:${REPO_ROOT}/src:${CLIENT_PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
for task in "${TASK_LIST[@]}"; do
  echo "[$(date --iso-8601=seconds)] starting ${task}: 25 val episodes, seed ${EVAL_SEED}"
  env \
    CUDA_VISIBLE_DEVICES= \
    DISPLAY="${DISPLAY_ID}" \
    COPPELIASIM_ROOT="${COPPELIASIM_ROOT}" \
    LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}" \
    QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}" \
    QT_QPA_PLATFORM=xcb \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${CLIENT_PYTHONPATH}" \
    "${RLBENCH_PY}" -m rlbench_worldpilot_wan_pi05.eval_online_rlbench_rpc_block12_per_event_v1 \
      --policy-url "http://${POLICY_HOST}:${POLICY_PORT}/infer" \
      --out "${OUT_DIR}/${task}.jsonl" \
      --manifest-path "${MANIFEST_PATH}" \
      --event-manifest-path "${EVENT_MANIFEST_PATH}" \
      --split val \
      --task "${task}" \
      --max-episodes-per-task 25 \
      --selection first \
      --seed "${EVAL_SEED}" \
      --wan-mode matched \
      --wan-seed-mode per_event \
      --wan-text-source task \
      --event-switch-mode pose_or_steps \
      --max-steps 30 \
      --max-steps-per-event 8 \
      --event-goal-pos-threshold 0.04 \
      --event-goal-rot-threshold 0.5 \
      --event-goal-gripper-threshold 0.5 \
      --rlbench-root "${RLBENCH_ROOT}" \
      --lowdim-root-200 "${LOWDIM_ROOT_200}" \
      --lowdim-root-400 "${LOWDIM_ROOT_400}" \
      --rgb-root-200 "${RGB_ROOT_200}" \
      --rgb-root-400 "${RGB_ROOT_400}" \
      --headless \
      >"${OUT_DIR}/${task}_eval.log" 2>&1 &
  CLIENT_PID=$!
  wait "${CLIENT_PID}"
  CLIENT_PID=
done

echo "[$(date --iso-8601=seconds)] completed Block12 per-event v1 seed ${EVAL_SEED}"
