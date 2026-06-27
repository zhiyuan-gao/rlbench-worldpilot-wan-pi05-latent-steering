# RLBench WorldPilot WAN pi0.5 Latent Steering

这是一个新的 sidecar repo，用来做 **WorldPilot-style Latent Steering + WAN future video latent + pi0.5 PyTorch** 实验。

它的定位和边界：

- 不修改 `rlbench_pi05_waypoint_baseline_20260606` 的 clean baseline 语义。
- 不 fork WorldPilot 作为主工程；`/raid/home/than/zhiyuan/WorldPilot` 只作为论文/代码 reference。
- 不 vendor OpenPI、Finetrainers/WAN 或 WorldPilot。
- 只做我们自己的 WAN latent provider、WorldPilot-style fuser、OpenPI/pi0.5 PyTorch steering glue、训练/eval 脚本。
- JAX 暂时不进入这个 repo；本 repo 只按 PyTorch/DDP 路线设计。

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
target:              next full-task heuristic waypoint
views:               front, left_shoulder, right_shoulder
language:            full-task instruction
```

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

## Shape Smoke

先只验证 WorldPilot-style WAN fuser 的 PyTorch shape：

```bash
bash scripts/smoke_fuser_shapes.sh
```

默认 smoke 使用 toy latent size，不代表实际 WAN latent 分辨率。真实训练/缓存时应以 WAN VAE-before-decode latent 的实际 shape 为准。

## WAN Latent Cache

先构建与 pi0.5 LeRobot samples 对齐的 sample index：

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
SPLIT=train \
bash scripts/export_wan_latent_cache.sh --resume
```

`wan-diffusers` backend 会读取当前三视角 RGB 和 target waypoint 三视角 RGB，hstack 后调用 WAN FLF pipeline，并保存 per-sample：

```text
future_video_latents: Tensor[V, C, T_lat, H_lat, W_lat]
latent_layout:        vcthw
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

Offline loss eval：

```bash
EXP_NAME=selected10_worldpilot_wan_pi05_torch \
bash scripts/eval_worldpilot_wan_pi05_torch.sh \
  --resume \
  --num-eval-batches 50
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
- LeRobot/sample-index alignment
- dummy and `wan-diffusers` WAN latent cache export
- latent cache validation
- WAN latent dataloader wrapper
- `PI0WanLatentSteeringPytorch`
- PyTorch/DDP train, save, resume
- offline eval loss entry

仍需要在 HPC 真机上验证：

- `wan-diffusers` backend 返回 latent 的实际 shape 是否和 diffusers 版本一致
- 真实 WAN LoRA 路径和显存配置
- 完整 online RLBench rollout eval
