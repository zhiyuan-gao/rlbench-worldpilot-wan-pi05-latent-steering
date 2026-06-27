#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

DEVICE="${DEVICE:-cpu}"
TIME_MODE="${WAN_LATENT_TIME_MODE:-all}"
LAYOUT="${WAN_LATENT_LAYOUT:-bvcthw}"

ARGS=(
  --device "${DEVICE}"
  --time-mode "${TIME_MODE}"
  --layout "${LAYOUT}"
  "$@"
)

if [[ -n "${PYTHON_BIN:-}" ]]; then
  "${PYTHON_BIN}" -m rlbench_worldpilot_wan_pi05.smoke_shapes "${ARGS[@]}"
elif [[ -d "${OPENPI_DIR}" ]] && command -v uv >/dev/null 2>&1; then
  cd "${OPENPI_DIR}"
  uv run python -m rlbench_worldpilot_wan_pi05.smoke_shapes "${ARGS[@]}"
else
  python3 -m rlbench_worldpilot_wan_pi05.smoke_shapes "${ARGS[@]}"
fi
