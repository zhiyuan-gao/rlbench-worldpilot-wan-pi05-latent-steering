#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

OPENVLA_OFT_GIT_URL="${OPENVLA_OFT_GIT_URL:-https://github.com/moojink/openvla-oft.git}"
OPENVLA_OFT_REF="${OPENVLA_OFT_REF:-main}"

mkdir -p "$(dirname "${OPENVLA_OFT_DIR}")" "${OPENVLA_OFT_CACHE_DIR}"

if [[ -d "${OPENVLA_OFT_DIR}/.git" ]]; then
  cd "${OPENVLA_OFT_DIR}"
  git fetch origin --prune
  git checkout "${OPENVLA_OFT_REF}"
  git pull --ff-only || true
else
  git clone -b "${OPENVLA_OFT_REF}" "${OPENVLA_OFT_GIT_URL}" "${OPENVLA_OFT_DIR}"
fi

cd "${OPENVLA_OFT_DIR}"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

cat <<EOF
OpenVLA-OFT checkout is ready:
  ${OPENVLA_OFT_DIR}

This does not download the 7B VLA yet. A Hugging Face model such as
${OPENVLA_OFT_VLA_PATH}
will be downloaded into HF_HOME/TRANSFORMERS_CACHE when a real model load runs.
EOF
