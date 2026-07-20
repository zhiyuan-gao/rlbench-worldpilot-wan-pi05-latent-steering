#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NPROC_PER_NODE=1 \
WANDB_ENABLED=0 \
bash "${REPO_ROOT}/scripts/train_fcrf_v1_hpc_4xh100.sh" \
  --smoke-only \
  --batch-size 1 \
  --num-workers 0 \
  --no-wandb-enabled \
  "$@"
