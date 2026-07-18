#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/hpc_paths.sh"
WORKSPACE="${WORKSPACE:-$(cd "${REPO_ROOT}/../.." && pwd)}"
export WORKSPACE

ALLOC_JOB_ID="${ALLOC_JOB_ID:-${SLURM_JOB_ID:-}}"
if [[ -z "${ALLOC_JOB_ID}" ]]; then
  echo "Set ALLOC_JOB_ID to an active 2-node allocation id, or run inside salloc." >&2
  exit 2
fi

NUM_NODES="${NUM_NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
if [[ "${NUM_NODES}" != "2" ]]; then
  echo "This launcher expects NUM_NODES=2: one node for WAN RPC and one node for policy/RLBench." >&2
  exit 2
fi

node_list="$(squeue -j "${ALLOC_JOB_ID}" -h -o "%N")"
readarray -t resolved_nodes < <(scontrol show hostnames "${node_list}")
WAN_NODE="${WAN_NODE:-${resolved_nodes[0]:-}}"
POLICY_NODE="${POLICY_NODE:-${resolved_nodes[1]:-}}"
if [[ -z "${WAN_NODE}" || -z "${POLICY_NODE}" ]]; then
  echo "Could not resolve two nodes from allocation ${ALLOC_JOB_ID}. Set WAN_NODE and POLICY_NODE manually." >&2
  exit 3
fi

OPENPI_PY="${OPENPI_PY:-${OPENPI_DIR}/.venv/bin/python}"
WAN_PY="${WAN_PY:-${WAN_EXPORT_PYTHON:-${OPENPI_PY}}}"
RLBENCH_PY="${RLBENCH_PY:-${WORKSPACE:-$(dirname "${REPO_ROOT}")}/.envs/rlbench/bin/python}"
if [[ ! -x "${OPENPI_PY}" ]]; then
  echo "OPENPI_PY is not executable: ${OPENPI_PY}" >&2
  exit 4
fi
if [[ ! -x "${WAN_PY}" ]]; then
  echo "WAN_PY is not executable: ${WAN_PY}" >&2
  exit 4
fi
if [[ ! -x "${RLBENCH_PY}" ]]; then
  echo "RLBENCH_PY is not executable: ${RLBENCH_PY}" >&2
  exit 4
fi

EXP_NAME="${EXP_NAME:-selected10_worldpilot_wan_pi05_block12_20260706}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-${CHECKPOINT_BASE_DIR}/pi05_rlbench_waypoint_h1/${EXP_NAME}/20000}"
WAN_LORA_DIR="${WAN_LORA_DIR:-${WAN_LORA:-}}"
WAN_NUM_FRAMES="${WAN_NUM_FRAMES:-21}"
WAN_NUM_INFERENCE_STEPS="${WAN_NUM_INFERENCE_STEPS:-${WAN_NUM_INFERENCE_STEPS_ONLINE:-1}}"
WAN_DTYPE="${WAN_DTYPE:-bf16}"
WAN_DEVICE_MAP="${WAN_DEVICE_MAP:-balanced}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda:0}"
POLICY_PORT="${POLICY_PORT:-18765}"
WAN_PORT="${WAN_PORT:-18766}"

SPLIT="${SPLIT:-val}"
TASK="${TASK:-ALL}"
MAX_EPISODES="${MAX_EPISODES:-}"
MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-25}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_STEPS_PER_EVENT="${MAX_STEPS_PER_EVENT:-8}"
XVFB_DISPLAY_NUM="${XVFB_DISPLAY_NUM:-188}"

task_tag="${TASK:-all}"
if [[ "${task_tag}" == "ALL" || "${task_tag}" == "all" ]]; then
  task_tag=all
fi
EVAL_TAG="${EVAL_TAG:-${EXP_NAME}_ckpt$(basename "${EVAL_CHECKPOINT}")_${SPLIT}_${task_tag}_2node}"

LOG_DIR="${LOG_DIR:-${WORKSPACE}/outputs/worldpilot_wan_pi05/logs}"
OUT_DIR="${OUT_DIR:-${WORKSPACE}/outputs/worldpilot_wan_pi05/online_eval}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
WAN_LOG="${LOG_DIR}/rpc_wan_${EVAL_TAG}_j${ALLOC_JOB_ID}_${STAMP}.log"
POLICY_LOG="${LOG_DIR}/rpc_policy_${EVAL_TAG}_j${ALLOC_JOB_ID}_${STAMP}.log"
CLIENT_LOG="${LOG_DIR}/rpc_client_${EVAL_TAG}_j${ALLOC_JOB_ID}_${STAMP}.log"
XVFB_LOG="${LOG_DIR}/xvfb_rpc_eval_${XVFB_DISPLAY_NUM}_${STAMP}.log"
OUT_JSONL="${OUT_DIR}/${EVAL_TAG}_${STAMP}.jsonl"
STOP_MARKER="${OUT_DIR}/${EVAL_TAG}_${STAMP}.stop"

echo "=== WorldPilot WAN pi0.5 online RPC eval ==="
echo "alloc: ${ALLOC_JOB_ID}"
echo "wan_node: ${WAN_NODE}"
echo "policy_rlbench_node: ${POLICY_NODE}"
echo "checkpoint: ${EVAL_CHECKPOINT}"
echo "wan_base: ${WAN_BASE_MODEL}"
echo "wan_lora: ${WAN_LORA_DIR:-<none>}"
echo "split/task: ${SPLIT}/${TASK}"
echo "max: max_episodes=${MAX_EPISODES:-none} max_per_task=${MAX_EPISODES_PER_TASK}"
echo "output: ${OUT_JSONL}"

srun --jobid "${ALLOC_JOB_ID}" \
  --nodes=2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  --nodelist="${WAN_NODE},${POLICY_NODE}" \
  --gres "gpu:${GPUS_PER_NODE}" \
  --kill-on-bad-exit=1 \
  bash -lc "
set -euo pipefail

source '${REPO_ROOT}/scripts/hpc_paths.sh'
export REPO_ROOT='${REPO_ROOT}'
export OPENPI_PY='${OPENPI_PY}'
export WAN_PY='${WAN_PY}'
export RLBENCH_PY='${RLBENCH_PY}'
export POLICY_PORT='${POLICY_PORT}'
export WAN_PORT='${WAN_PORT}'
export POLICY_DEVICE='${POLICY_DEVICE}'
export WAN_NODE='${WAN_NODE}'
export POLICY_NODE='${POLICY_NODE}'
export WAN_LOG='${WAN_LOG}'
export POLICY_LOG='${POLICY_LOG}'
export CLIENT_LOG='${CLIENT_LOG}'
export XVFB_LOG='${XVFB_LOG}'
export OUT_JSONL='${OUT_JSONL}'
export STOP_MARKER='${STOP_MARKER}'
export SPLIT='${SPLIT}'
export TASK='${TASK}'
export MAX_EPISODES='${MAX_EPISODES}'
export MAX_EPISODES_PER_TASK='${MAX_EPISODES_PER_TASK}'
export MAX_STEPS='${MAX_STEPS}'
export MAX_STEPS_PER_EVENT='${MAX_STEPS_PER_EVENT}'
export XVFB_DISPLAY_NUM='${XVFB_DISPLAY_NUM}'

export PYTHONPATH=\"\${REPO_ROOT}/src:\${OPENPI_DIR}/src:\${RLBENCH_ROOT}:\${PYTHONPATH:-}\"
export HF_HOME=\"\${HF_HOME:-\${WORKSPACE}/.cache/huggingface}\"
export COPPELIASIM_ROOT='${COPPELIASIM_ROOT}'
export LD_LIBRARY_PATH=\"\${COPPELIASIM_ROOT}:\${COPPELIASIM_ROOT}/lib:\${LD_LIBRARY_PATH:-}\"
export PATH=\"\${COPPELIASIM_ROOT}:\${PATH}\"
export QT_QPA_PLATFORM_PLUGIN_PATH=\"\${COPPELIASIM_ROOT}\"

HOSTSHORT=\$(hostname -s)
TASK_ARGS=()
if [[ -n \"\${TASK}\" && \"\${TASK}\" != \"ALL\" && \"\${TASK}\" != \"all\" ]]; then
  TASK_ARGS=(--task \"\${TASK}\")
fi
MAX_EPISODES_ARGS=()
if [[ -n \"\${MAX_EPISODES}\" ]]; then
  MAX_EPISODES_ARGS=(--max-episodes \"\${MAX_EPISODES}\")
fi

health_check() {
  local host=\$1
  local port=\$2
  \"\${OPENPI_PY}\" - <<'PY' \"\${host}\" \"\${port}\" >/dev/null 2>&1
import pickle
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f\"http://{host}:{port}/health\", timeout=5) as response:
    payload = pickle.loads(response.read())
if not payload.get(\"ok\"):
    raise SystemExit(1)
PY
}

if [[ \"\${HOSTSHORT}\" == \"\${WAN_NODE}\" ]]; then
  rm -f \"\${STOP_MARKER}\"
  WAN_LORA_ARGS=()
  if [[ -n '${WAN_LORA_DIR}' ]]; then
    WAN_LORA_ARGS=(--wan-lora-dir '${WAN_LORA_DIR}')
  fi
  CUDA_VISIBLE_DEVICES=0,1,2,3 \"\${WAN_PY}\" -m rlbench_worldpilot_wan_pi05.serve_wan_latent_rpc \
    --host 0.0.0.0 \
    --port \"\${WAN_PORT}\" \
    --wan-base-model '${WAN_BASE_MODEL}' \
    \"\${WAN_LORA_ARGS[@]}\" \
    --wan-num-frames '${WAN_NUM_FRAMES}' \
    --wan-num-inference-steps '${WAN_NUM_INFERENCE_STEPS}' \
    --wan-output-layout '${WAN_OUTPUT_LAYOUT}' \
    --wan-dtype '${WAN_DTYPE}' \
    --wan-device-map '${WAN_DEVICE_MAP}' \
    >\"\${WAN_LOG}\" 2>&1 &
  WAN_PID=\$!
  echo \"[\${HOSTSHORT}] Started WAN RPC server pid=\${WAN_PID}\" | tee -a \"\${WAN_LOG}\"
  while kill -0 \"\${WAN_PID}\" 2>/dev/null; do
    if [[ -f \"\${STOP_MARKER}\" ]]; then
      kill \"\${WAN_PID}\" 2>/dev/null || true
      wait \"\${WAN_PID}\" 2>/dev/null || true
      exit 0
    fi
    sleep 5
  done
  wait \"\${WAN_PID}\"
  exit \$?
fi

if [[ \"\${HOSTSHORT}\" != \"\${POLICY_NODE}\" ]]; then
  echo \"Unexpected host \${HOSTSHORT}; expected \${WAN_NODE} or \${POLICY_NODE}\" >&2
  exit 2
fi

cleanup() {
  touch \"\${STOP_MARKER}\" 2>/dev/null || true
  if [[ -n \"\${XVFB_PID:-}\" ]] && kill -0 \"\${XVFB_PID}\" 2>/dev/null; then
    kill \"\${XVFB_PID}\" 2>/dev/null || true
    wait \"\${XVFB_PID}\" 2>/dev/null || true
  fi
  if [[ -n \"\${POLICY_PID:-}\" ]] && kill -0 \"\${POLICY_PID}\" 2>/dev/null; then
    kill \"\${POLICY_PID}\" 2>/dev/null || true
    wait \"\${POLICY_PID}\" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for i in \$(seq 1 120); do
  if health_check \"\${WAN_NODE}\" \"\${WAN_PORT}\"; then
    echo \"[\${HOSTSHORT}] WAN RPC server is healthy\"
    break
  fi
  sleep 10
  if [[ \"\${i}\" == 120 ]]; then
    echo \"Timed out waiting for WAN RPC server; tail follows:\" >&2
    tail -n 160 \"\${WAN_LOG}\" >&2 || true
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES=0,1,2,3 \"\${OPENPI_PY}\" -m rlbench_worldpilot_wan_pi05.serve_policy_rpc \
  pi05_rlbench_waypoint_h1 \
  --host 127.0.0.1 \
  --port \"\${POLICY_PORT}\" \
  --exp-name '${EXP_NAME}' \
  --eval-checkpoint '${EVAL_CHECKPOINT}' \
  --checkpoint-base-dir '${CHECKPOINT_BASE_DIR}' \
  --assets-base-dir '${ASSETS_BASE_DIR}' \
  --policy-device \"\${POLICY_DEVICE}\" \
  --pytorch-training-precision bfloat16 \
  --wan-latent-shape '${WAN_LATENT_SHAPE}' \
  --wan-steering-mode '${WAN_STEERING_MODE}' \
  --wan-steering-block '${WAN_STEERING_BLOCK}' \
  --wan-steering-gate '${WAN_STEERING_GATE}' \
  --action-num-steps 10 \
  --wan-rpc-url \"http://\${WAN_NODE}:\${WAN_PORT}/latent\" \
  >\"\${POLICY_LOG}\" 2>&1 &
POLICY_PID=\$!
echo \"[\${HOSTSHORT}] Started policy RPC server pid=\${POLICY_PID}\"

for i in \$(seq 1 120); do
  if health_check 127.0.0.1 \"\${POLICY_PORT}\"; then
    echo \"[\${HOSTSHORT}] Policy RPC server is healthy\"
    break
  fi
  if ! kill -0 \"\${POLICY_PID}\" 2>/dev/null; then
    echo \"Policy server exited early; tail follows:\" >&2
    tail -n 160 \"\${POLICY_LOG}\" >&2 || true
    exit 1
  fi
  sleep 10
  if [[ \"\${i}\" == 120 ]]; then
    echo \"Timed out waiting for policy RPC server; tail follows:\" >&2
    tail -n 160 \"\${POLICY_LOG}\" >&2 || true
    exit 1
  fi
done

export DISPLAY=\":\${XVFB_DISPLAY_NUM}\"
export QT_QPA_PLATFORM=xcb
Xvfb \"\${DISPLAY}\" -screen 0 1280x1024x24 -nolisten tcp >\"\${XVFB_LOG}\" 2>&1 &
XVFB_PID=\$!
sleep 2

CUDA_VISIBLE_DEVICES= \"\${RLBENCH_PY}\" -m rlbench_worldpilot_wan_pi05.eval_online_rlbench_rpc \
  --policy-url \"http://127.0.0.1:\${POLICY_PORT}/infer\" \
  --out \"\${OUT_JSONL}\" \
  --manifest-path \"\${MANIFEST_PATH}\" \
  --event-manifest-path \"\${EVENT_MANIFEST_PATH}\" \
  --split \"\${SPLIT}\" \
  \"\${TASK_ARGS[@]}\" \
  \"\${MAX_EPISODES_ARGS[@]}\" \
  --max-episodes-per-task \"\${MAX_EPISODES_PER_TASK}\" \
  --max-steps \"\${MAX_STEPS}\" \
  --max-steps-per-event \"\${MAX_STEPS_PER_EVENT}\" \
  --rlbench-root \"\${RLBENCH_ROOT}\" \
  --lowdim-root-200 \"\${LOWDIM_ROOT_200}\" \
  --lowdim-root-400 \"\${LOWDIM_ROOT_400}\" \
  --rgb-root-200 \"\${RGB_ROOT_200}\" \
  --rgb-root-400 \"\${RGB_ROOT_400}\" \
  --headless \
  >\"\${CLIENT_LOG}\" 2>&1

cat \"\${OUT_JSONL%.jsonl}.summary.json\"
"
