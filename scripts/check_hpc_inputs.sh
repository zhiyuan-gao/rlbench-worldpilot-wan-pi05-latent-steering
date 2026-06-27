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

missing=0
check_path "OPENPI_DIR" "${OPENPI_DIR}" || missing=1
check_path "PI05_BASELINE_REPO" "${PI05_BASELINE_REPO}" || missing=1
check_path "WORLDPILOT_DIR" "${WORLDPILOT_DIR}" || missing=1
check_path "FINETRAINERS_DIR" "${FINETRAINERS_DIR}" || missing=1
check_path "SELECTED1500_ROOT" "${SELECTED1500_ROOT}" || missing=1
check_path "MANIFEST_PATH" "${MANIFEST_PATH}" || missing=1

if [[ -e "${WAN_LATENT_CACHE_ROOT}" ]]; then
  printf '[OK]   WAN_LATENT_CACHE_ROOT: %s\n' "${WAN_LATENT_CACHE_ROOT}"
else
  printf '[WARN] WAN_LATENT_CACHE_ROOT not created yet: %s\n' "${WAN_LATENT_CACHE_ROOT}"
fi

printf '\nLeRobot repo id: %s\n' "${LEROBOT_REPO_ID}"
printf 'HF_LEROBOT_HOME: %s\n' "${HF_LEROBOT_HOME}"

exit "${missing}"

