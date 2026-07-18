#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/hpc_paths.sh"

ALLOC_JOB_ID="${ALLOC_JOB_ID:-${SLURM_JOB_ID:-}}"
if [[ -z "${ALLOC_JOB_ID}" ]]; then
  echo "Set ALLOC_JOB_ID to an active 2-node allocation id, or run inside salloc." >&2
  exit 2
fi

NUM_NODES="${NUM_NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE}}"
WORLD_SIZE="$((NUM_NODES * GPUS_PER_NODE))"
MASTER_PORT="${MASTER_PORT:-29537}"
RDZV_ID="${RDZV_ID:-${ALLOC_JOB_ID}_worldpilot_full_event}"

EXP_NAME="${EXP_NAME:-selected10_worldpilot_wan_pi05_full_event_2node8_1step_contact002500}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
KEEP_PERIOD="${KEEP_PERIOD:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PEAK_LR="${PEAK_LR:-}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-wan_fuser}"

CACHE_GPUS_PER_SHARD="${CACHE_GPUS_PER_SHARD:-2}"
CACHE_SHARDS="${CACHE_SHARDS:-$((WORLD_SIZE / CACHE_GPUS_PER_SHARD))}"
CACHE_TASKS_PER_NODE="${CACHE_TASKS_PER_NODE:-$((GPUS_PER_NODE / CACHE_GPUS_PER_SHARD))}"
EXPORT_CACHE="${EXPORT_CACHE:-1}"
VALIDATE_CACHE="${VALIDATE_CACHE:-1}"
CONVERT_PI05="${CONVERT_PI05:-1}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"

ensure_openpi_transformers_replace() {
  local openpi_python="${OPENPI_DIR}/.venv/bin/python"
  local replace_src="${OPENPI_DIR}/src/openpi/models_pytorch/transformers_replace"
  if [[ ! -x "${openpi_python}" ]]; then
    echo "Missing OpenPI venv python: ${openpi_python}" >&2
    exit 5
  fi
  if [[ ! -d "${replace_src}" ]]; then
    echo "Missing OpenPI transformers_replace source: ${replace_src}" >&2
    exit 5
  fi

  local transformers_version
  transformers_version="$("${openpi_python}" - <<'PY'
import transformers
print(transformers.__version__)
PY
)"
  if [[ "${transformers_version}" != "4.53.2" ]]; then
    echo "Installing transformers==4.53.2 into OpenPI .venv (found ${transformers_version})"
    cd "${OPENPI_DIR}"
    uv pip install --python "${openpi_python}" "transformers==4.53.2"
  fi

  local transformers_dir
  transformers_dir="$("${openpi_python}" - <<'PY'
import pathlib
import transformers
print(pathlib.Path(transformers.__file__).resolve().parent)
PY
)"
  echo "Installing OpenPI transformers_replace into ${transformers_dir}"
  cp -r "${replace_src}/." "${transformers_dir}/"

  "${openpi_python}" - <<'PY'
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
print("OpenPI transformers_replace check: OK")
PY
}

ensure_wan_export_python() {
  local wan_python="${WAN_EXPORT_PYTHON:-${OPENPI_DIR}/.venv/bin/python}"
  if [[ ! -x "${wan_python}" ]]; then
    echo "WAN_EXPORT_PYTHON is not executable: ${wan_python}" >&2
    exit 5
  fi

  "${wan_python}" - <<'PY'
from pathlib import Path
import importlib.metadata as md
import diffusers

pipeline_path = Path(diffusers.__file__).resolve().parent / "pipelines" / "wan" / "pipeline_wan_i2v.py"
if "last_image" not in pipeline_path.read_text():
    raise RuntimeError(f"WAN export Python uses a WanImageToVideoPipeline without last_image: {pipeline_path}")

print(
    "WAN export Python OK:",
    "diffusers=" + md.version("diffusers"),
    "accelerate=" + md.version("accelerate"),
    "peft=" + md.version("peft"),
)
PY
}

mkdir -p \
  "${WAN_LATENT_CACHE_ROOT}" \
  "${CHECKPOINT_BASE_DIR}" \
  "${ASSETS_BASE_DIR}" \
  "${WANDB_DIR}" \
  "${WORKSPACE}/outputs/worldpilot_wan_pi05/logs"

echo "=== WorldPilot WAN pi0.5 Full Event training ==="
echo "repo: ${REPO_ROOT}"
echo "alloc: ${ALLOC_JOB_ID}"
echo "exp: ${EXP_NAME}"
echo "world_size: ${WORLD_SIZE} (${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs)"
echo "cache_export: ${CACHE_SHARDS} shards x ${CACHE_GPUS_PER_SHARD} GPUs/shard"
echo "wan cache: ${WAN_LATENT_CACHE_ROOT}"
echo "wan lora: ${WAN_LORA_DIR}"
echo "pytorch init: ${PYTORCH_WEIGHT_PATH}"
echo "trainable_scope: ${TRAINABLE_SCOPE}"
echo "wan_steering: mode=${WAN_STEERING_MODE} block=${WAN_STEERING_BLOCK} gate=${WAN_STEERING_GATE}"

cd "${REPO_ROOT}"
if [[ "${EXPORT_CACHE}" == "1" ]]; then
  REQUIRE_WAN_BASE_MODEL=1 bash scripts/check_hpc_inputs.sh
else
  REQUIRE_WAN_BASE_MODEL=0 bash scripts/check_hpc_inputs.sh
fi

echo "=== Build sample indexes ==="
SPLIT=train bash scripts/build_sample_index.sh
SPLIT=val bash scripts/build_sample_index.sh

if [[ "${CONVERT_PI05}" == "1" && ! -f "${PYTORCH_WEIGHT_PATH}/model.safetensors" ]]; then
  echo "=== Convert pi0.5 JAX checkpoint to PyTorch ==="
  ensure_openpi_transformers_replace
  cd "${OPENPI_DIR}"
  PYTHONUNBUFFERED=1 .venv/bin/python examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir "${JAX_PI05_CHECKPOINT}" \
    --config-name pi05_rlbench_waypoint_h1 \
    --output-path "${PYTORCH_WEIGHT_PATH}" \
    --precision bfloat16
fi

if [[ ! -f "${PYTORCH_WEIGHT_PATH}/model.safetensors" ]]; then
  echo "Missing PyTorch init model: ${PYTORCH_WEIGHT_PATH}/model.safetensors" >&2
  exit 3
fi

if [[ "${EXPORT_CACHE}" == "1" ]]; then
  ensure_wan_export_python

  echo "=== Export WAN latent cache: train split, ${CACHE_SHARDS} shards ==="
  srun --jobid "${ALLOC_JOB_ID}" \
    --nodes "${NUM_NODES}" \
    --ntasks "${CACHE_SHARDS}" \
    --ntasks-per-node "${CACHE_TASKS_PER_NODE}" \
    --cpus-per-task 8 \
    --kill-on-bad-exit=1 \
    bash -lc "
set -euo pipefail
source '${REPO_ROOT}/scripts/hpc_paths.sh'
first_gpu=\$((SLURM_LOCALID * ${CACHE_GPUS_PER_SHARD}))
last_gpu=\$((first_gpu + ${CACHE_GPUS_PER_SHARD} - 1))
export CUDA_VISIBLE_DEVICES=\"\${first_gpu}\"
for gpu in \$(seq \$((first_gpu + 1)) \${last_gpu}); do
  export CUDA_VISIBLE_DEVICES=\"\${CUDA_VISIBLE_DEVICES},\${gpu}\"
done
cd '${REPO_ROOT}'
WAN_LATENT_BACKEND=wan-diffusers \
WAN_EXPECTED_BACKEND=wan-diffusers \
SPLIT=train \
bash scripts/export_wan_latent_cache.sh \
  --num-shards '${CACHE_SHARDS}' \
  --shard-index \"\${SLURM_PROCID}\" \
  --resume
" 2>&1 | tee "${WORKSPACE}/outputs/worldpilot_wan_pi05/logs/export_train_${CACHE_SHARDS}shards.log"

  echo "=== Export WAN latent cache: val split, ${CACHE_SHARDS} shards ==="
  srun --jobid "${ALLOC_JOB_ID}" \
    --nodes "${NUM_NODES}" \
    --ntasks "${CACHE_SHARDS}" \
    --ntasks-per-node "${CACHE_TASKS_PER_NODE}" \
    --cpus-per-task 8 \
    --kill-on-bad-exit=1 \
    bash -lc "
set -euo pipefail
source '${REPO_ROOT}/scripts/hpc_paths.sh'
first_gpu=\$((SLURM_LOCALID * ${CACHE_GPUS_PER_SHARD}))
last_gpu=\$((first_gpu + ${CACHE_GPUS_PER_SHARD} - 1))
export CUDA_VISIBLE_DEVICES=\"\${first_gpu}\"
for gpu in \$(seq \$((first_gpu + 1)) \${last_gpu}); do
  export CUDA_VISIBLE_DEVICES=\"\${CUDA_VISIBLE_DEVICES},\${gpu}\"
done
cd '${REPO_ROOT}'
WAN_LATENT_BACKEND=wan-diffusers \
WAN_EXPECTED_BACKEND=wan-diffusers \
SPLIT=val \
bash scripts/export_wan_latent_cache.sh \
  --num-shards '${CACHE_SHARDS}' \
  --shard-index \"\${SLURM_PROCID}\" \
  --resume
" 2>&1 | tee "${WORKSPACE}/outputs/worldpilot_wan_pi05/logs/export_val_${CACHE_SHARDS}shards.log"
fi

if [[ "${VALIDATE_CACHE}" == "1" ]]; then
  echo "=== Validate WAN latent cache ==="
  WAN_EXPECTED_BACKEND=wan-diffusers SPLIT=train bash scripts/validate_wan_latent_cache.sh
  WAN_EXPECTED_BACKEND=wan-diffusers SPLIT=val bash scripts/validate_wan_latent_cache.sh
fi

node_list="$(squeue -j "${ALLOC_JOB_ID}" -h -o "%N")"
MASTER_ADDR="$(scontrol show hostnames "${node_list}" | head -n 1)"
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "Could not resolve MASTER_ADDR from allocation ${ALLOC_JOB_ID}" >&2
  exit 4
fi

train_extra=()
if [[ "${RESUME}" == "1" ]]; then
  train_extra+=(--resume)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  train_extra+=(--overwrite)
fi
if [[ -n "${PEAK_LR}" ]]; then
  train_extra+=(--lr-schedule.peak-lr "${PEAK_LR}")
fi

echo "=== Launch DDP train ==="
srun --jobid "${ALLOC_JOB_ID}" \
  --nodes "${NUM_NODES}" \
  --ntasks "${NUM_NODES}" \
  --ntasks-per-node 1 \
  --gres "gpu:${GPUS_PER_NODE}" \
  --cpus-per-task 32 \
  --kill-on-bad-exit=1 \
  --label \
  bash -lc "
set -euo pipefail
source '${REPO_ROOT}/scripts/hpc_paths.sh'
export EXP_NAME='${EXP_NAME}'
export WANDB_ENABLED='${WANDB_ENABLED}'
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
cd '${OPENPI_DIR}'
uv run torchrun \
  --nnodes='${NUM_NODES}' \
  --nproc_per_node='${NPROC_PER_NODE}' \
  --node_rank=\"\${SLURM_PROCID}\" \
  --rdzv_id='${RDZV_ID}' \
  --rdzv_backend=c10d \
  --rdzv_endpoint='${MASTER_ADDR}:${MASTER_PORT}' \
  -m rlbench_worldpilot_wan_pi05.train_torch \
  pi05_rlbench_waypoint_h1 \
  --exp-name '${EXP_NAME}' \
  --lerobot-repo-id '${LEROBOT_REPO_ID}' \
  --manifest-path '${MANIFEST_PATH}' \
  --event-manifest-path '${EVENT_MANIFEST_PATH}' \
  --goal-mode '${WAN_LATENT_GOAL_MODE}' \
  --sample-index-path '${WAN_LATENT_CACHE_ROOT}/sample_index_train.jsonl' \
  --wan-latent-cache-root '${WAN_LATENT_CACHE_ROOT}' \
  --split train \
  --pytorch-weight-path '${PYTORCH_WEIGHT_PATH}' \
  --checkpoint-base-dir '${CHECKPOINT_BASE_DIR}' \
  --assets-base-dir '${ASSETS_BASE_DIR}' \
  --trainable-scope '${TRAINABLE_SCOPE}' \
  --wan-steering-mode '${WAN_STEERING_MODE}' \
  --wan-steering-block '${WAN_STEERING_BLOCK}' \
  --wan-steering-gate '${WAN_STEERING_GATE}' \
  --expected-wan-num-inference-steps '${WAN_NUM_INFERENCE_STEPS}' \
  --expected-wan-backend '${WAN_EXPECTED_BACKEND}' \
  --expected-wan-latent-shape '${WAN_LATENT_SHAPE}' \
  --batch-size '${BATCH_SIZE}' \
  --num-train-steps '${NUM_TRAIN_STEPS}' \
  --num-workers '${NUM_WORKERS}' \
  --save-interval '${SAVE_INTERVAL}' \
  --keep-period '${KEEP_PERIOD}' \
  --lr-schedule.warmup-steps '${WARMUP_STEPS}' \
  --no-wandb-enabled \
  ${train_extra[*]}
"

echo "Done. Checkpoints: ${CHECKPOINT_BASE_DIR}/pi05_rlbench_waypoint_h1/${EXP_NAME}"
