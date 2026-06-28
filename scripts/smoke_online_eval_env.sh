#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

: "${OPENPI_DIR:?Set OPENPI_DIR or source scripts/setup_env.sh}"
: "${COPPELIASIM_ROOT:?Set COPPELIASIM_ROOT before running online eval smoke.}"

if [[ ! -d "${COPPELIASIM_ROOT}" ]]; then
  echo "COPPELIASIM_ROOT does not exist: ${COPPELIASIM_ROOT}" >&2
  exit 1
fi

case ":${PYTHONPATH:-}:" in
  *":${RLBENCH_ROOT}:"*) ;;
  *) export PYTHONPATH="${RLBENCH_ROOT}:${PYTHONPATH:-}" ;;
esac

export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${COPPELIASIM_ROOT}/lib:${LD_LIBRARY_PATH:-}"

cd "${OPENPI_DIR}"
uv run python - <<'PY'
import importlib
import inspect
import os

print("OPENPI_DIR=", os.environ.get("OPENPI_DIR"))
print("RLBENCH_ROOT=", os.environ.get("RLBENCH_ROOT"))
print("COPPELIASIM_ROOT=", os.environ.get("COPPELIASIM_ROOT"))

for name in ("torch", "openpi", "pyrep", "rlbench", "diffusers"):
    module = importlib.import_module(name)
    path = getattr(module, "__file__", "<namespace>")
    print(f"import {name}: OK ({path})")

root = os.environ["COPPELIASIM_ROOT"]
required = ["libcoppeliaSim.so", "coppeliaSim.sh"]
missing = [item for item in required if not os.path.exists(os.path.join(root, item))]
if missing:
    raise SystemExit(f"COPPELIASIM_ROOT is missing expected files: {missing}")

from diffusers import WanImageToVideoPipeline

supports_last_image = "last_image" in inspect.signature(WanImageToVideoPipeline.__call__).parameters
print("WanImageToVideoPipeline supports last_image:", supports_last_image)
if not supports_last_image:
    raise SystemExit("This Python environment cannot run real FLF WAN online eval; use a patched diffusers/WAN env.")

print("Online eval environment smoke looks OK.")
PY
