#!/usr/bin/env bash
# Copy this file to scripts/hpc_paths.sh on the HPC, edit the paths there,
# then source scripts/hpc_paths.sh before running this repo's scripts.
#
# The main scripts call scripts/setup_env.sh internally. setup_env.sh respects
# variables that are already exported here, so these values override the local
# defaults without editing source code.

# Main HPC workspace. In the common case, edit only this path and place data,
# models, caches, checkpoints, and repos under the derived layout below.
export HPC_ROOT=/scratch/$USER/rlbench_worldpilot_wan_pi05

# These empty workspace/output directories can be created automatically.
mkdir -p \
  "${HPC_ROOT}/repos" \
  "${HPC_ROOT}/data" \
  "${HPC_ROOT}/models" \
  "${HPC_ROOT}/cache" \
  "${HPC_ROOT}/checkpoints" \
  "${HPC_ROOT}/assets" \
  "${HPC_ROOT}/sim" \
  "${HPC_ROOT}/wandb"

# This repo.
export REPO_ROOT=${HPC_ROOT}/repos/rlbench_worldpilot_wan_pi05_latent_steering_20260628

# OpenPI / pi0.5 baseline side.
export PI05_ROOT=${HPC_ROOT}/pi05_baseline
export OPENPI_DIR=${PI05_ROOT}/openpi
export PI05_BASELINE_REPO=${HPC_ROOT}/repos/rlbench_pi05_waypoint_baseline_20260606
export HF_LEROBOT_HOME=${PI05_ROOT}/lerobot_home
export LEROBOT_REPO_ID=rlbench/selected10_pi05_waypoint_h1

# Raw selected1500 data. This must contain the actual dataset; mkdir above only
# creates the parent directory. Override this variable if the dataset lives on
# another shared filesystem.
export SELECTED1500_DATASET_ROOT=${HPC_ROOT}/data/selected1500_dataset
export RGB_ROOT_200=${SELECTED1500_DATASET_ROOT}/local200/rgb3_keyframes_intervals
export RGB_ROOT_400=${SELECTED1500_DATASET_ROOT}/remote400/rgb3_keyframes_intervals
export LOWDIM_ROOT_200=${SELECTED1500_DATASET_ROOT}/local200/nonimage_metadata
export LOWDIM_ROOT_400=${SELECTED1500_DATASET_ROOT}/remote400/nonimage_metadata

# Manifests.
export MANIFEST_PATH=${PI05_BASELINE_REPO}/manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl
export EVENT_MANIFEST_PATH=${SELECTED1500_DATASET_ROOT}/manifests/selected10_event_fullinfo_train100_val25_test25_from_train450_stratified_20260606.jsonl
export WAN_LATENT_GOAL_MODE=event_end

# WAN model and generated latent cache. WAN_BASE_MODEL and WAN_LORA_DIR are
# input resources; put them under HPC_ROOT/models or override these paths.
export WAN_BASE_MODEL=${HPC_ROOT}/models/Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers
export WAN_LORA_DIR=${HPC_ROOT}/models/wan_lora
export WAN_LATENT_CACHE_ROOT=${HPC_ROOT}/cache/selected10_worldpilot_wan_latent_cache
export WAN_NUM_INFERENCE_STEPS=1
export WAN_OUTPUT_LAYOUT=bcthw
export WAN_LATENT_SHAPE=3,16,6,32,32
export WAN_EXPECTED_BACKEND=wan-diffusers

# PyTorch pi0.5 checkpoint and outputs.
export PYTORCH_WEIGHT_PATH=${HPC_ROOT}/checkpoints/pi05_pytorch_checkpoint
export CHECKPOINT_BASE_DIR=${HPC_ROOT}/checkpoints/worldpilot_wan_pi05_checkpoints
export ASSETS_BASE_DIR=${HPC_ROOT}/assets/worldpilot_wan_pi05_assets
export WANDB_DIR=${HPC_ROOT}/wandb

mkdir -p "${WAN_LATENT_CACHE_ROOT}" "${CHECKPOINT_BASE_DIR}" "${ASSETS_BASE_DIR}" "${WANDB_DIR}"

# Optional OpenVLA-OFT route. The repo checkout is a code dependency; the
# checkpoint can be either a local path or a Hugging Face repo id. It is not
# downloaded by this path file or by shape smoke tests.
export OPENVLA_OFT_DIR=${HPC_ROOT}/repos/openvla-oft
export OPENVLA_OFT_VLA_PATH=openvla/openvla-7b
export OPENVLA_OFT_CHECKPOINT=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
export OPENVLA_OFT_CACHE_DIR=${HPC_ROOT}/cache/openvla_oft_hf
export OPENVLA_OFT_NUM_ACTIONS_CHUNK=8
export OPENVLA_OFT_ACTION_DIM=7
export OPENVLA_OFT_PROPRIO_DIM=7
export OPENVLA_OFT_NUM_IMAGES_IN_INPUT=3
export OPENVLA_OFT_STATS_PATH=${HPC_ROOT}/cache/openvla_oft_rlbench_dataset_statistics.json
export OPENVLA_OFT_RUN_ROOT=${HPC_ROOT}/checkpoints/openvla_oft_rlbench

# Online RLBench eval only.
export RLBENCH_ROOT=${HPC_ROOT}/repos/RLBench
export COPPELIASIM_ROOT=${HPC_ROOT}/sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
