#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

SPLIT="${SPLIT:-train}" \
NPROC_PER_NODE=1 \
bash "${REPO_ROOT}/scripts/train_openvla_oft_rlbench.sh" \
  --dry-run \
  --dry-run-samples "${OPENVLA_OFT_DRY_RUN_SAMPLES:-4}" \
  "$@"
