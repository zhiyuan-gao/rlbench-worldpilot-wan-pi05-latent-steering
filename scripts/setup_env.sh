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
export RLBENCH_ROOT="${RLBENCH_ROOT:-/raid/home/than/zhiyuan/RLBench}"
export SELECTED1500_DATASET_ROOT="${SELECTED1500_DATASET_ROOT:-/raid/home/than/zhiyuan/selected1500_dataset}"
export SELECTED1500_ROOT="${SELECTED1500_ROOT:-${SELECTED1500_DATASET_ROOT}}"
export RGB_ROOT_200="${RGB_ROOT_200:-${SELECTED1500_DATASET_ROOT}/local200/rgb3_keyframes_intervals}"
export RGB_ROOT_400="${RGB_ROOT_400:-${SELECTED1500_DATASET_ROOT}/remote400/rgb3_keyframes_intervals}"
export LOWDIM_ROOT_200="${LOWDIM_ROOT_200:-${SELECTED1500_DATASET_ROOT}/local200/nonimage_metadata}"
export LOWDIM_ROOT_400="${LOWDIM_ROOT_400:-${SELECTED1500_DATASET_ROOT}/remote400/nonimage_metadata}"

export LEROBOT_REPO_ID="${LEROBOT_REPO_ID:-rlbench/selected10_pi05_waypoint_h1}"
export MANIFEST_PATH="${MANIFEST_PATH:-${PI05_BASELINE_REPO}/manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl}"
export EVENT_MANIFEST_PATH="${EVENT_MANIFEST_PATH:-${SELECTED1500_DATASET_ROOT}/manifests/selected10_event_fullinfo_train100_val25_test25_from_train450_stratified_20260606.jsonl}"
export WAN_LATENT_GOAL_MODE="${WAN_LATENT_GOAL_MODE:-event_end}"
export WAN_LATENT_CACHE_ROOT="${WAN_LATENT_CACHE_ROOT:-${REPO_ROOT}/latent_cache/selected10_worldpilot_wan_pi05}"
export WAN_BASE_MODEL="${WAN_BASE_MODEL:-/raid/home/than/zhiyuan/finetrainers/pretrained_models/Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers}"
export WAN_LORA_DIR="${WAN_LORA_DIR:-}"
export WAN_NUM_INFERENCE_STEPS="${WAN_NUM_INFERENCE_STEPS:-1}"
export WAN_OUTPUT_LAYOUT="${WAN_OUTPUT_LAYOUT:-bcthw}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${PI05_ROOT}/openpi_cache}"
export HF_HOME="${HF_HOME:-${PI05_ROOT}/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI05_ROOT}/lerobot_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PI05_ROOT}/uv_cache}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"

case ":${PYTHONPATH:-}:" in
  *":${REPO_ROOT}/src:"*) ;;
  *) export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" ;;
esac
