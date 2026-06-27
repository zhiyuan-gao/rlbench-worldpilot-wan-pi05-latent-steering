#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

SPLIT="${SPLIT:-train}"
SAMPLE_INDEX_PATH="${SAMPLE_INDEX_PATH:-${WAN_LATENT_CACHE_ROOT}/sample_index_${SPLIT}.jsonl}"

cd "${OPENPI_DIR}"
uv run python -m rlbench_worldpilot_wan_pi05.sample_index \
  --manifest-path "${MANIFEST_PATH}" \
  --out "${SAMPLE_INDEX_PATH}" \
  --split "${SPLIT}" \
  --sample-every-n "${SAMPLE_EVERY_N:-0}" \
  --rgb-root-200 "${RGB_ROOT_200}" \
  --rgb-root-400 "${RGB_ROOT_400}" \
  "$@"

