# RLBench WorldPilot WAN pi0.5 Latent Steering

这个 repo 用来探索 **WorldPilot-style Latent Steering + WAN future video latent + pi0.5 PyTorch** 的 RLBench 动作模型实验。

核心想法来自 WorldPilot：先用 world model 预测未来视觉 latent，再把这些 future latent 作为额外的 scene prior 注入到 VLA/VLM policy 的 hidden states 里，帮助动作模型在当前观测之外利用“接下来应该到哪里”的视觉信息。本 repo 把这个机制迁移到我们的 RLBench setting：用训练好的 WAN 三视角 future-video model 产生 VAE-before-decode latent，再通过 WorldPilot-style cross-attention fuser 接到 PyTorch pi0.5 上训练动作策略。

代码结构上，这里主要包含三部分：

- WAN future video latent 的 sample 对齐、导出、缓存和校验。
- WorldPilot-style `WanFutureVideoFuser`，把 `(V, C, T_lat, H_lat, W_lat)` future latent 变成可注入 pi0.5 的 hidden tokens。
- PyTorch pi0.5 latent-steering 训练/eval 入口，沿用 RLBench pi0.5 waypoint baseline 的 LeRobot 数据格式和 action target。

## Branch Profile

当前 `hpc-4xh100-nvl` 分支对应 **4x H100 NVL 94GB** 配置。

```text
branch             hpc-4xh100-nvl
trainer            PyTorch / DDP
default GPUs       4x H100 NVL 94GB
NPROC_PER_NODE     4
pi0.5 baseline     rlbench_pi05_waypoint_baseline_20260606 hpc-4xh100-nvl branch
dataset format     same LeRobot dataset as pi0.5 waypoint baseline
```

另一个分支应保持为：

```text
main               8x A100 40GB, NPROC_PER_NODE=8
```

两条分支只应该在 HPC profile、默认 GPU 数、显存建议和 README 示例上不同；数据格式和方法代码保持一致。

## Method Contract

目标不是复刻 WorldPilot 的工程，而是对齐它的 **Latent Steering 机制**：

```text
VLM hidden states:          (B, L, H_pi05)
future scene latent tokens: (B, N_latent_tokens, H_pi05)
cross-attn residual:        (B, L, H_pi05)
```

WorldPilot 公开代码里，Cosmos future image latent 通常是：

```text
(B, N_cam, C, H_lat, W_lat)
```

然后每个 camera latent 被 flatten/project 成一个 VLM hidden token。我们的 WAN 版本更接近 Cosmos-Predict future-video latent 的方式：

```text
WAN VAE-before-decode future video latent:
  preferred cache shape: (B, V, C, T_lat, H_lat, W_lat)

WorldPilot-style WAN fuser:
  (B, V, C, T_lat, H_lat, W_lat)
  -> select/preserve latent time
  -> (B, V * K, C * H_lat * W_lat)
  -> Linear(..., H_pi05)
  -> cross-attn into pi0.5 VLM hidden states
```

默认研究路线先用 `time_mode=all`，也就是保留 WAN latent-time 维度，把每个 `(view, latent_time)` 作为一个 future-scene token。后续可以做 ablation：

- `all`: 保留所有 latent time tokens，最接近 future-video latent steering。
- `last`: 每个 view 只取最后一个 latent time token，更接近“目标/终点 latent”。
- `mean`: 对 latent time 做平均，token 数更少但信息压缩更强。

这个 repo 不使用 Wan transformer block13 hidden tokens；Latent Steering 对齐的是 **VAE-before-decode future video latent**。

## Current Experiment Semantics

本实验把 pi0.5 的动作监督和 WAN steering cache 明确分开：

```text
pi0.5 action supervision:
  current frame -> next full-task heuristic waypoint

WAN future latent cache:
  current frame -> current event/subgoal end frame
  denoise depth: 1-step WAN denoising latent, before VAE decode
```

例如某条 episode 的 full-task waypoints 是 `[36, 49, 61, 92, 108]`，event/subgoal ends 是 `[49, 108]`，那么：

```text
current=0   action_target=36   wan_latent_goal=49
current=36  action_target=49   wan_latent_goal=49
current=49  action_target=61   wan_latent_goal=108
current=61  action_target=92   wan_latent_goal=108
current=92  action_target=108  wan_latent_goal=108
```

所以本 repo 的 pi0.5 action target 仍然和原版 RLBench pi0.5 waypoint baseline 一致；WAN cache 的 target image/latent 改成当前 event/subgoal end，并默认保存 **1-step denoise 后、VAE decode 前** 的 WAN future latent。

## Dataset Format

HPC 数据格式沿用 pi0.5 waypoint baseline repo。

默认 LeRobot dataset：

```text
repo_id: rlbench/selected10_pi05_waypoint_h1
```

来源 repo：

```text
/raid/home/than/zhiyuan/corl2026/rlbench_pi05_waypoint_baseline_20260606
```

默认 manifest：

```text
/raid/home/than/zhiyuan/corl2026/rlbench_pi05_waypoint_baseline_20260606/manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl
```

raw selected1500 数据默认路径：

```text
SELECTED1500_DATASET_ROOT=/raid/home/than/zhiyuan/selected1500_dataset
RGB_ROOT_200=${SELECTED1500_DATASET_ROOT}/local200/rgb3_keyframes_intervals
RGB_ROOT_400=${SELECTED1500_DATASET_ROOT}/remote400/rgb3_keyframes_intervals
LOWDIM_ROOT_200=${SELECTED1500_DATASET_ROOT}/local200/nonimage_metadata
LOWDIM_ROOT_400=${SELECTED1500_DATASET_ROOT}/remote400/nonimage_metadata
```

LeRobot 数据仍由 pi0.5 baseline repo 的 conversion 脚本生成，默认读取：

```text
HF_LEROBOT_HOME=${PI05_ROOT}/lerobot_home
repo_id=rlbench/selected10_pi05_waypoint_h1
```

动作和 proprio 不重新定义：

```text
state/action format: absolute_rotvec7 = x, y, z, rx, ry, rz, gripper_open
action_horizon:      1
action target:       next full-task heuristic waypoint
WAN latent target:   current event/subgoal end frame
WAN denoise depth:   1-step, VAE-before-decode latent
views:               front, left_shoulder, right_shoulder
language:            full-task instruction
```

也就是说，动作监督仍然和原版 RLBench pi0.5 waypoint baseline 一致；新增的 WAN steering latent 才看向当前 sample 所属 event 的 end frame。

本 repo 新增的只是 WAN latent cache。建议 cache 以 LeRobot sample / original RLBench frame 对齐，至少包含：

```text
future_video_latents: float16/bfloat16 tensor, preferred shape (V, C, T_lat, H_lat, W_lat)
view_names:           ["front", "left_shoulder", "right_shoulder"]
task
variation
episode
source_bundle
frame_index
target_waypoint_frame
latent_goal_frame
event_idx
event_end_frame
instruction
wan_checkpoint_or_run_id
latent_layout
```

更详细的约定见 [docs/data_format.md](docs/data_format.md)。

## Install

进入 repo：

```bash
cd /raid/home/than/zhiyuan/corl2026/rlbench_worldpilot_wan_pi05_latent_steering_20260628
pip install -e . --no-deps
pip install -r requirements.txt
```

加载默认路径：

```bash
source scripts/setup_env.sh
```

检查 HPC 输入路径：

```bash
bash scripts/check_hpc_inputs.sh
```

## HPC Step-by-Step

这一节是把 repo 复制到 HPC 后，从路径配置到正式训练的完整顺序。这个 repo 是 sidecar repo：它不重新生成 pi0.5 LeRobot 数据，也不 vendor OpenPI/WAN，只在 pi0.5 baseline 数据旁边新增 WAN future-video latent cache 和 PyTorch latent steering 训练入口。

### 0. Select Branch

8x A100 机器使用：

```bash
cd /path/to/rlbench_worldpilot_wan_pi05_latent_steering_20260628
git checkout main
```

4x H100 NVL 机器使用：

```bash
cd /path/to/rlbench_worldpilot_wan_pi05_latent_steering_20260628
git checkout hpc-4xh100-nvl
```

### 1. Configure Required Paths

建议在 Slurm 脚本里用 `export` 覆盖路径，而不是反复改源码。至少需要这些路径：

```bash
export REPO_ROOT=/path/to/rlbench_worldpilot_wan_pi05_latent_steering_20260628

export PI05_ROOT=/path/to/pi05_baseline
export OPENPI_DIR=${PI05_ROOT}/openpi
export PI05_BASELINE_REPO=/path/to/rlbench_pi05_waypoint_baseline_20260606

export SELECTED1500_DATASET_ROOT=/path/to/selected1500_dataset
export RGB_ROOT_200=${SELECTED1500_DATASET_ROOT}/local200/rgb3_keyframes_intervals
export RGB_ROOT_400=${SELECTED1500_DATASET_ROOT}/remote400/rgb3_keyframes_intervals
export LOWDIM_ROOT_200=${SELECTED1500_DATASET_ROOT}/local200/nonimage_metadata
export LOWDIM_ROOT_400=${SELECTED1500_DATASET_ROOT}/remote400/nonimage_metadata

export MANIFEST_PATH=${PI05_BASELINE_REPO}/manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl
export EVENT_MANIFEST_PATH=${SELECTED1500_DATASET_ROOT}/manifests/selected10_event_fullinfo_train100_val25_test25_from_train450_stratified_20260606.jsonl
export WAN_LATENT_GOAL_MODE=event_end

export HF_LEROBOT_HOME=${PI05_ROOT}/lerobot_home
export LEROBOT_REPO_ID=rlbench/selected10_pi05_waypoint_h1
```

其中 `HF_LEROBOT_HOME` 必须能找到 pi0.5 baseline 已经转换好的 LeRobot 数据：

```text
${HF_LEROBOT_HOME}/rlbench/selected10_pi05_waypoint_h1/meta/info.json
```

如果这个文件不存在，需要先回到 `rlbench_pi05_waypoint_baseline_20260606` 按 baseline 流程生成 LeRobot dataset。

### 2. Configure WAN Latent Cache Paths

WAN latent cache 是本 repo 新增的数据。建议放在 scratch 或大容量共享盘：

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-FLF2V-14B-720P-diffusers
export WAN_LORA_DIR=/path/to/trained_wan_lora
export WAN_LATENT_CACHE_ROOT=/scratch/path/selected10_worldpilot_wan_latent_cache
export WAN_NUM_INFERENCE_STEPS=1
export RLBENCH_ROOT=/path/to/RLBench
```

如果只是先做 pipeline smoke test，可以让 LoRA 为空：

```bash
export WAN_LORA_DIR=
```

`WAN_LATENT_CACHE_ROOT` 下会生成：

```text
sample_index_train.jsonl
sample_index_val.jsonl
...
task/variation/episode/frame_xxx.pt
```

默认 `WAN_NUM_INFERENCE_STEPS=1`，也就是保存 WAN 经过一步 denoising 后、VAE decode 前的 future-video latent。若要跑 3-step/5-step 消融，可以覆盖这个环境变量或给 export 脚本传 `--num-inference-steps`。

如果之前已经用其他 denoise step 数导出过 cache，不要和当前 1-step 实验混用；建议换一个新的 `WAN_LATENT_CACHE_ROOT` 或用 `--overwrite` 重新导出。

每个 `.pt` 文件里主要保存：

```text
future_video_latents: Tensor[V, C, T_lat, H_lat, W_lat]
latent_layout:        vcthw
view_names:           ["front", "left_shoulder", "right_shoulder"]
```

### 3. Check Inputs

```bash
cd ${REPO_ROOT}
source scripts/setup_env.sh
bash scripts/check_hpc_inputs.sh
```

需要重点确认这些项是 OK：

```text
OPENPI_DIR
PI05_BASELINE_REPO
SELECTED1500_DATASET_ROOT
RGB_ROOT_200
RGB_ROOT_400
LOWDIM_ROOT_200
LOWDIM_ROOT_400
MANIFEST_PATH
EVENT_MANIFEST_PATH
WAN_BASE_MODEL
LeRobot dataset
```

如果 `LeRobot dataset` 是 WARN，说明 pi0.5 baseline 数据没有准备好，后面的训练 dataloader 会失败。

### 4. Build Sample Index

sample index 把 pi0.5 LeRobot sample 对齐回原始 RLBench episode/frame/waypoint，并为每个 sample 记录两套 target：

```text
action_target_waypoint_frame = next full-task heuristic waypoint
latent_goal_frame            = current event/subgoal end frame
```

```bash
SPLIT=train bash scripts/build_sample_index.sh
SPLIT=val bash scripts/build_sample_index.sh
```

默认输出：

```text
${WAN_LATENT_CACHE_ROOT}/sample_index_train.jsonl
${WAN_LATENT_CACHE_ROOT}/sample_index_val.jsonl
```

### 5. Dummy Cache Smoke Test

先不要跑真实 WAN，先用 dummy latent 确认 sample index、cache 读取和 OpenPI dataloader 能连起来：

```bash
WAN_LATENT_BACKEND=dummy SPLIT=train \
bash scripts/export_wan_latent_cache.sh --max-samples 16 --overwrite

SPLIT=train \
bash scripts/validate_wan_latent_cache.sh --max-samples 16
```

`--max-samples` 只限制本次导出的 cache 数量，不会把 `sample_index_train.jsonl` 截短。`validate_wan_latent_cache.sh` 会同时检查 cache 里的 `metadata.num_inference_steps` 是否等于当前 `WAN_NUM_INFERENCE_STEPS`。

然后做训练 dataloader dry-run：

```bash
export PYTORCH_WEIGHT_PATH=/path/to/pi05_pytorch_checkpoint

SPLIT=train \
WAN_NUM_INFERENCE_STEPS=1 \
NPROC_PER_NODE=4 \
bash scripts/train_worldpilot_wan_pi05_torch.sh \
  --dry-run \
  --allow-missing-latents \
  --batch-size 4 \
  --no-wandb-enabled
```

8x A100 分支把 `NPROC_PER_NODE` 改成 `8`。

`--dry-run` 不启动训练，也不加载 pi0.5 base weights；它用于检查 transformed LeRobot dataset、sample index、latent cache batch 和 split/step 配置。fuser shape 单测仍然用 `bash scripts/smoke_fuser_shapes.sh`。

### 6. Export Real WAN Latent Cache

dummy smoke test 通过后，导出真实 WAN VAE-before-decode future video latent：

```bash
WAN_LATENT_BACKEND=wan-diffusers WAN_NUM_INFERENCE_STEPS=1 SPLIT=train \
bash scripts/export_wan_latent_cache.sh --resume

WAN_LATENT_BACKEND=wan-diffusers WAN_NUM_INFERENCE_STEPS=1 SPLIT=val \
bash scripts/export_wan_latent_cache.sh --resume
```

数据量大时可以用 shard 并行。例如 8 个 Slurm jobs：

```bash
WAN_LATENT_BACKEND=wan-diffusers WAN_NUM_INFERENCE_STEPS=1 SPLIT=train \
bash scripts/export_wan_latent_cache.sh \
  --num-shards 8 \
  --shard-index 0 \
  --resume
```

其他 jobs 分别使用 `--shard-index 1` 到 `--shard-index 7`。

导出后检查 coverage 和 shape：

```bash
SPLIT=train bash scripts/validate_wan_latent_cache.sh
SPLIT=val bash scripts/validate_wan_latent_cache.sh
```

### 7. Prepare pi0.5 PyTorch Weights

训练脚本需要 PyTorch 版 pi0.5 checkpoint：

```bash
export PYTORCH_WEIGHT_PATH=/path/to/pi05_pytorch_checkpoint
```

如果只有 JAX checkpoint，先在 OpenPI 里转换：

```bash
cd ${OPENPI_DIR}
uv run examples/convert_jax_model_to_pytorch.py \
  --config-name pi05_rlbench_waypoint_h1 \
  --checkpoint-dir /path/to/jax/pi05_checkpoint \
  --output-path /path/to/pi05_pytorch_checkpoint
```

### 8. Train

4x H100 NVL:

```bash
cd ${REPO_ROOT}

export EXP_NAME=selected10_worldpilot_wan_pi05_torch
export CHECKPOINT_BASE_DIR=/scratch/path/worldpilot_wan_pi05_checkpoints
export WANDB_DIR=/scratch/path/wandb

NPROC_PER_NODE=4 \
bash scripts/train_worldpilot_wan_pi05_torch.sh \
  --batch-size 128 \
  --num-train-steps 20000 \
  --save-interval 2000 \
  --keep-period 2000 \
  --lr-schedule.warmup-steps 10000
```

8x A100:

```bash
NPROC_PER_NODE=8 \
bash scripts/train_worldpilot_wan_pi05_torch.sh \
  --batch-size 128 \
  --num-train-steps 20000 \
  --save-interval 2000 \
  --keep-period 2000 \
  --lr-schedule.warmup-steps 10000
```

最短路径清单：

```text
REPO_ROOT                 new WorldPilot-WAN-pi0.5 repo
OPENPI_DIR                pi0.5 OpenPI repo
PI05_BASELINE_REPO        original pi0.5 baseline repo
SELECTED1500_DATASET_ROOT raw selected1500 dataset
MANIFEST_PATH             waypoint manifest
EVENT_MANIFEST_PATH       event/subgoal manifest
HF_LEROBOT_HOME           converted pi0.5 LeRobot data home
WAN_BASE_MODEL            WAN FLF base model
WAN_LORA_DIR              trained WAN video-model LoRA
WAN_LATENT_CACHE_ROOT     generated WAN latent cache
WAN_NUM_INFERENCE_STEPS   default 1 for 1-step denoise WAN latent
PYTORCH_WEIGHT_PATH       pi0.5 PyTorch checkpoint
CHECKPOINT_BASE_DIR       output directory for this experiment
```

### 9. Online RLBench Rollout Eval

Online eval 不读取预先导出的 WAN latent cache；它会在 RLBench 闭环里实时跑 WAN：

```text
live RLBench obs + current-event goal RGB + text
  -> 1-step WAN FLF latent before VAE decode
  -> PyTorch pi0.5 latent-steering policy
  -> absolute_rotvec7 waypoint
  -> rotvec-to-quaternion RLBench planner action
```

动作语义仍然是原版 pi0.5 baseline 的 full-task next waypoint。WAN steering latent 的 goal 是当前 event/subgoal end；同一个 event 内 goal image 固定，每个控制步用新的当前三视角 RGB 重新跑 WAN。切换到下一个 event 后才换新的 event-end goal image。

先检查 online eval 环境：

```bash
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
bash scripts/smoke_online_eval_env.sh
```

如果想直接用训练目录下最新 checkpoint：

```bash
cd ${REPO_ROOT}

export SPLIT=val
export EXP_NAME=selected10_worldpilot_wan_pi05_torch
export CHECKPOINT_BASE_DIR=/scratch/path/worldpilot_wan_pi05_checkpoints
export ONLINE_EVAL_OUT=/scratch/path/worldpilot_wan_pi05_online/${EXP_NAME}_${SPLIT}.jsonl

WAN_LATENT_BACKEND=wan-diffusers \
WAN_NUM_INFERENCE_STEPS=1 \
bash scripts/eval_online_rlbench_worldpilot_wan_pi05_torch.sh \
  --task meat_off_grill \
  --max-episodes-per-task 25 \
  --max-steps 30 \
  --max-steps-per-event 8
```

如果要指定某个 step：

```bash
export EVAL_CHECKPOINT=${CHECKPOINT_BASE_DIR}/pi05_rlbench_waypoint_h1/${EXP_NAME}/20000
bash scripts/eval_online_rlbench_worldpilot_wan_pi05_torch.sh --task meat_off_grill
```

输出包含逐 episode JSONL 和 summary：

```text
${ONLINE_EVAL_OUT}
${ONLINE_EVAL_OUT%.jsonl}.summary.json
```

可以先用 dummy WAN latent 做 plumbing smoke：

```bash
WAN_LATENT_BACKEND=dummy \
bash scripts/eval_online_rlbench_worldpilot_wan_pi05_torch.sh \
  --task meat_off_grill \
  --max-episodes 1 \
  --max-steps 1
```

`wan-diffusers` online backend 需要当前环境的 `WanImageToVideoPipeline` 支持 FLF 的 `last_image` 参数；如果当前 diffusers 版本不支持，需要切到 WAN/Finetrainers 对应环境或 patched pipeline。

## Shape Smoke

先只验证 WorldPilot-style WAN fuser 的 PyTorch shape：

```bash
bash scripts/smoke_fuser_shapes.sh
```

默认 smoke 使用 toy latent size，不代表实际 WAN latent 分辨率。真实训练/缓存时应以 WAN VAE-before-decode latent 的实际 shape 为准。

## WAN Latent Cache

先构建与 pi0.5 LeRobot samples 对齐的 sample index。默认 `WAN_LATENT_GOAL_MODE=event_end`，所以 action target 仍是 next waypoint，WAN latent goal 是当前 event/subgoal end：

```bash
SPLIT=train \
bash scripts/build_sample_index.sh
```

快速 pipeline smoke 可以先导出 dummy latent cache：

```bash
WAN_LATENT_BACKEND=dummy \
SPLIT=train \
bash scripts/export_wan_latent_cache.sh --max-samples 16 --overwrite
```

真实 WAN VAE-before-decode latent cache：

```bash
WAN_LATENT_BACKEND=wan-diffusers \
WAN_BASE_MODEL=/raid/home/than/zhiyuan/finetrainers/pretrained_models/Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers \
WAN_LORA_DIR=/path/to/wan_lora \
WAN_NUM_INFERENCE_STEPS=1 \
SPLIT=train \
bash scripts/export_wan_latent_cache.sh --resume
```

`wan-diffusers` backend 会读取当前三视角 RGB 和当前 event/subgoal end 的三视角 RGB，hstack 后调用 WAN FLF pipeline，并保存 per-sample 的 1-step denoise pre-VAE latent：

```text
future_video_latents: Tensor[V, C, T_lat, H_lat, W_lat]
latent_layout:        vcthw
metadata.num_inference_steps: 1
```

导出后检查 coverage 和 shape：

```bash
SPLIT=train \
bash scripts/validate_wan_latent_cache.sh
```

## Training Entry

训练入口先固定为 PyTorch/DDP：

```bash
PI05_ROOT=/raid/home/than/zhiyuan/corl2026/pi05_baseline \
PYTORCH_WEIGHT_PATH=/path/to/pytorch/pi05_base \
WAN_LATENT_CACHE_ROOT=/path/to/wan_latent_cache \
EXP_NAME=selected10_worldpilot_wan_pi05_torch \
NPROC_PER_NODE=4 \
bash scripts/train_worldpilot_wan_pi05_torch.sh --dry-run
```

`--dry-run` 会构建 OpenPI transformed LeRobot dataset、sample index 和 WAN latent batch，但不启动训练。正式训练去掉 `--dry-run`：

```bash
PI05_ROOT=/raid/home/than/zhiyuan/corl2026/pi05_baseline \
PYTORCH_WEIGHT_PATH=/path/to/pytorch/pi05_base \
EXP_NAME=selected10_worldpilot_wan_pi05_torch \
NPROC_PER_NODE=4 \
bash scripts/train_worldpilot_wan_pi05_torch.sh \
  --batch-size 128 \
  --num-train-steps 20000 \
  --save-interval 2000 \
  --keep-period 2000 \
  --lr-schedule.warmup-steps 10000
```

Resume：

```bash
EXP_NAME=selected10_worldpilot_wan_pi05_torch \
NPROC_PER_NODE=4 \
bash scripts/train_worldpilot_wan_pi05_torch.sh --resume
```

Offline loss eval 默认也按当前 LeRobot repo 对齐的 split 读取 sample index；如果 `HF_LEROBOT_HOME` 里只有 train split 转换结果，就保持 `SPLIT=train`。如果单独准备了 val LeRobot repo/cache，再显式设 `SPLIT=val`：

```bash
EXP_NAME=selected10_worldpilot_wan_pi05_torch \
SPLIT=train \
bash scripts/eval_worldpilot_wan_pi05_torch.sh \
  --resume \
  --num-eval-batches 50
```

Online RLBench rollout eval：

```bash
EXP_NAME=selected10_worldpilot_wan_pi05_torch \
CHECKPOINT_BASE_DIR=/scratch/path/worldpilot_wan_pi05_checkpoints \
WAN_LATENT_BACKEND=wan-diffusers \
WAN_NUM_INFERENCE_STEPS=1 \
SPLIT=val \
bash scripts/eval_online_rlbench_worldpilot_wan_pi05_torch.sh \
  --task meat_off_grill \
  --max-episodes-per-task 25
```

训练脚本现在会实例化 `PI0WanLatentSteeringPytorch`，在 OpenPI PyTorch pi0.5 的 `embed_prefix()` 后注入 `WanFutureVideoFuser`，然后保持 action head、action target、state/action format 不变。

如果只有 JAX pi0.5 checkpoint，需要先在 OpenPI 里转换为 PyTorch 权重：

```bash
cd /raid/home/than/zhiyuan/corl2026/pi05_baseline/openpi
uv run examples/convert_jax_model_to_pytorch.py \
  --config-name pi05_rlbench_waypoint_h1 \
  --checkpoint-dir /path/to/jax/pi05_base_or_checkpoint \
  --output-path /path/to/pytorch/pi05_base
```

## Implementation Status

已实现：

- raw selected1500 path profile
- LeRobot/sample-index alignment with unchanged next-waypoint action targets
- event-end WAN latent goals for current-event/subgoal steering
- dummy and `wan-diffusers` WAN latent cache export
- latent cache validation
- WAN latent dataloader wrapper
- `PI0WanLatentSteeringPytorch`
- PyTorch/DDP train, save, resume
- offline eval loss entry
- online RLBench rollout eval entry with event/subgoal goal scheduling and per-step WAN latent refresh

仍需要在 HPC 真机上验证：

- `wan-diffusers` backend 返回 latent 的实际 shape 是否和 diffusers 版本一致
- 真实 WAN LoRA 路径和显存配置
- online RLBench rollout 的 CoppeliaSim/RLBench 运行、WAN online backend 显存配置和成功率
