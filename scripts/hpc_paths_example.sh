#!/usr/bin/env bash
# Copy this file to scripts/hpc_paths.sh on the HPC, edit the paths there,
# then source scripts/hpc_paths.sh before running this repo's scripts.
#
# The main scripts call scripts/setup_env.sh internally. setup_env.sh respects
# variables that are already exported here, so these values override the local
# defaults without editing source code.

# Workspace and this repo.
export WORKSPACE=/path/to/workspace
export REPO_ROOT=${WORKSPACE}/baselines/rlbench_worldpilot_wan_pi05_latent_steering_20260628

# OpenPI / pi0.5 baseline side.
export PI05_ROOT=/path/to/pi05_baseline
export OPENPI_DIR=${PI05_ROOT}/openpi
export PI05_BASELINE_REPO=/path/to/rlbench_pi05_waypoint_baseline_20260606
export HF_LEROBOT_HOME=${PI05_ROOT}/lerobot_home
export LEROBOT_REPO_ID=rlbench/selected10_pi05_waypoint_h1

# Raw selected1500 data.
export SELECTED1500_DATASET_ROOT=/path/to/selected1500_dataset
export RGB_ROOT_200=${SELECTED1500_DATASET_ROOT}/local200/rgb3_keyframes_intervals
export RGB_ROOT_400=${SELECTED1500_DATASET_ROOT}/remote400/rgb3_keyframes_intervals
export LOWDIM_ROOT_200=${SELECTED1500_DATASET_ROOT}/local200/nonimage_metadata
export LOWDIM_ROOT_400=${SELECTED1500_DATASET_ROOT}/remote400/nonimage_metadata

# Manifests.
export MANIFEST_PATH=${PI05_BASELINE_REPO}/manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl
export EVENT_MANIFEST_PATH=${SELECTED1500_DATASET_ROOT}/manifests/selected10_event_fullinfo_train100_val25_test25_from_train450_stratified_20260606.jsonl
export WAN_LATENT_GOAL_MODE=event_end

# WAN model and generated latent cache.
export WAN_BASE_MODEL=/path/to/Wan2.1-FLF2V-14B-720P-diffusers
export WAN_LORA_DIR=/path/to/trained_wan_lora
export WAN_EXPORT_PYTHON=/path/to/finetrainers-wan/bin/python
export WAN_LATENT_CACHE_ROOT=/scratch/path/selected10_worldpilot_wan_latent_cache
export WAN_NUM_INFERENCE_STEPS=1
export WAN_OUTPUT_LAYOUT=bcthw
export WAN_LATENT_SHAPE=3,16,6,32,32
export WAN_EXPECTED_BACKEND=wan-diffusers
export WAN_STEERING_MODE=early   # early or block
export WAN_STEERING_BLOCK=12
export WAN_STEERING_GATE=auto

# PyTorch pi0.5 checkpoint and outputs.
export PYTORCH_WEIGHT_PATH=/path/to/pi05_pytorch_checkpoint
export CHECKPOINT_BASE_DIR=/scratch/path/worldpilot_wan_pi05_checkpoints
export ASSETS_BASE_DIR=/scratch/path/worldpilot_wan_pi05_assets
export WANDB_DIR=/scratch/path/wandb

# Online RLBench eval only.
export RLBENCH_ROOT=/path/to/RLBench
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
