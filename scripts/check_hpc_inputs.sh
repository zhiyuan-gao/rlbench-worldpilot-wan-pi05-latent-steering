#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    printf '[OK]   %s: %s\n' "${label}" "${path}"
  else
    printf '[MISS] %s: %s\n' "${label}" "${path}"
    return 1
  fi
}

check_optional_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    printf '[OK]   %s: %s\n' "${label}" "${path}"
  else
    printf '[WARN] %s not found, only needed for reference/debugging: %s\n' "${label}" "${path}"
  fi
}

missing=0
check_path "OPENPI_DIR" "${OPENPI_DIR}" || missing=1
check_path "PI05_BASELINE_REPO" "${PI05_BASELINE_REPO}" || missing=1
check_optional_path "WORLDPILOT_DIR" "${WORLDPILOT_DIR}"
check_optional_path "FINETRAINERS_DIR" "${FINETRAINERS_DIR}"
check_path "SELECTED1500_DATASET_ROOT" "${SELECTED1500_DATASET_ROOT}" || missing=1
check_path "RGB_ROOT_200" "${RGB_ROOT_200}" || missing=1
check_path "RGB_ROOT_400" "${RGB_ROOT_400}" || missing=1
check_path "LOWDIM_ROOT_200" "${LOWDIM_ROOT_200}" || missing=1
check_path "LOWDIM_ROOT_400" "${LOWDIM_ROOT_400}" || missing=1
check_path "MANIFEST_PATH" "${MANIFEST_PATH}" || missing=1
check_path "EVENT_MANIFEST_PATH" "${EVENT_MANIFEST_PATH}" || missing=1
if [[ "${REQUIRE_WAN_BASE_MODEL:-1}" == "1" ]]; then
  check_path "WAN_BASE_MODEL" "${WAN_BASE_MODEL}" || missing=1
else
  check_optional_path "WAN_BASE_MODEL" "${WAN_BASE_MODEL}"
fi

if [[ -e "${WAN_LATENT_CACHE_ROOT}" ]]; then
  printf '[OK]   WAN_LATENT_CACHE_ROOT: %s\n' "${WAN_LATENT_CACHE_ROOT}"
else
  printf '[WARN] WAN_LATENT_CACHE_ROOT not created yet: %s\n' "${WAN_LATENT_CACHE_ROOT}"
fi

printf '\nLeRobot repo id: %s\n' "${LEROBOT_REPO_ID}"
printf 'HF_LEROBOT_HOME: %s\n' "${HF_LEROBOT_HOME}"
if [[ -e "${HF_LEROBOT_HOME}/${LEROBOT_REPO_ID}/meta/info.json" ]]; then
  printf '[OK]   LeRobot dataset: %s\n' "${HF_LEROBOT_HOME}/${LEROBOT_REPO_ID}"
else
  printf '[WARN] LeRobot dataset not found yet: %s\n' "${HF_LEROBOT_HOME}/${LEROBOT_REPO_ID}"
  printf '       Run the pi0.5 baseline conversion first or point HF_LEROBOT_HOME to the converted dataset.\n'
fi

exit "${missing}"
