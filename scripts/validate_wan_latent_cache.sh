#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

SPLIT="${SPLIT:-train}"
SAMPLE_INDEX_PATH="${SAMPLE_INDEX_PATH:-${WAN_LATENT_CACHE_ROOT}/sample_index_${SPLIT}.jsonl}"

cd "${OPENPI_DIR}"
uv run python -m rlbench_worldpilot_wan_pi05.validate_cache \
  --sample-index-path "${SAMPLE_INDEX_PATH}" \
  --cache-root "${WAN_LATENT_CACHE_ROOT}" \
  --expected-num-inference-steps "${WAN_NUM_INFERENCE_STEPS}" \
  "$@"
