#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

missing=0

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    printf '[OK]   %s: %s\n' "${label}" "${path}"
  else
    printf '[MISS] %s: %s\n' "${label}" "${path}"
    missing=1
  fi
}

printf 'OpenVLA-OFT repo/checkpoint configuration\n'
printf 'OPENVLA_OFT_DIR=%s\n' "${OPENVLA_OFT_DIR}"
printf 'OPENVLA_OFT_VLA_PATH=%s\n' "${OPENVLA_OFT_VLA_PATH}"
printf 'OPENVLA_OFT_CHECKPOINT=%s\n' "${OPENVLA_OFT_CHECKPOINT}"
printf 'OPENVLA_OFT_CACHE_DIR=%s\n' "${OPENVLA_OFT_CACHE_DIR}"
printf 'OPENVLA_OFT_NUM_IMAGES_IN_INPUT=%s\n' "${OPENVLA_OFT_NUM_IMAGES_IN_INPUT}"
printf 'OPENVLA_OFT_PROPRIO_DIM=%s\n' "${OPENVLA_OFT_PROPRIO_DIM}"
printf '\n'

check_path "OPENVLA_OFT_DIR" "${OPENVLA_OFT_DIR}"
check_path "openvla_utils.py" "${OPENVLA_OFT_DIR}/experiments/robot/openvla_utils.py"
check_path "modeling_prismatic.py" "${OPENVLA_OFT_DIR}/prismatic/extern/hf/modeling_prismatic.py"

if [[ -e "${OPENVLA_OFT_CHECKPOINT}" ]]; then
  printf '[OK]   OPENVLA_OFT_CHECKPOINT local path: %s\n' "${OPENVLA_OFT_CHECKPOINT}"
elif [[ "${OPENVLA_OFT_CHECKPOINT}" == */* ]]; then
  printf '[OK]   OPENVLA_OFT_CHECKPOINT looks like a Hugging Face repo id: %s\n' "${OPENVLA_OFT_CHECKPOINT}"
  printf '       It will be downloaded/cached only when real OpenVLA-OFT loading runs.\n'
else
  printf '[MISS] OPENVLA_OFT_CHECKPOINT is neither a local path nor a HF repo id: %s\n' "${OPENVLA_OFT_CHECKPOINT}"
  missing=1
fi

if [[ -e "${OPENVLA_OFT_VLA_PATH}" ]]; then
  printf '[OK]   OPENVLA_OFT_VLA_PATH local path: %s\n' "${OPENVLA_OFT_VLA_PATH}"
elif [[ "${OPENVLA_OFT_VLA_PATH}" == */* ]]; then
  printf '[OK]   OPENVLA_OFT_VLA_PATH looks like a Hugging Face repo id: %s\n' "${OPENVLA_OFT_VLA_PATH}"
  printf '       It will be downloaded/cached only when real OpenVLA-OFT model loading runs.\n'
else
  printf '[WARN] OPENVLA_OFT_VLA_PATH is not a local path or HF-style repo id: %s\n' "${OPENVLA_OFT_VLA_PATH}"
fi

exit "${missing}"
