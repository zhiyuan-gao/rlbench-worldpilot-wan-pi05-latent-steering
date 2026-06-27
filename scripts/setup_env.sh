#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export REPO_ROOT
export PI05_ROOT="${PI05_ROOT:-/raid/home/than/zhiyuan/corl2026/pi05_baseline}"
export OPENPI_DIR="${OPENPI_DIR:-${PI05_ROOT}/openpi}"
export PI05_BASELINE_REPO="${PI05_BASELINE_REPO:-/raid/home/than/zhiyuan/corl2026/rlbench_pi05_waypoint_baseline_20260606}"
export WORLDPILOT_DIR="${WORLDPILOT_DIR:-/raid/home/than/zhiyuan/WorldPilot}"
export FINETRAINERS_DIR="${FINETRAINERS_DIR:-/raid/home/than/zhiyuan/finetrainers}"
export SELECTED1500_ROOT="${SELECTED1500_ROOT:-/raid/home/than/zhiyuan/selected1500_dataset}"

export LEROBOT_REPO_ID="${LEROBOT_REPO_ID:-rlbench/selected10_pi05_waypoint_h1}"
export MANIFEST_PATH="${MANIFEST_PATH:-${PI05_BASELINE_REPO}/manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl}"
export WAN_LATENT_CACHE_ROOT="${WAN_LATENT_CACHE_ROOT:-${REPO_ROOT}/latent_cache/selected10_worldpilot_wan_pi05}"

export HF_HOME="${HF_HOME:-/raid/home/than/zhiyuan/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HF_HOME}/openpi}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"

case ":${PYTHONPATH:-}:" in
  *":${REPO_ROOT}/src:"*) ;;
  *) export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" ;;
esac

