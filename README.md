# RLBench WorldPilot WAN pi0.5 Latent Steering

这个 repo 用来探索 **WorldPilot-style Latent Steering + WAN future video latent + pi0.5 PyTorch** 的 RLBench 动作模型实验。

核心想法来自 WorldPilot：先用 world model 预测未来视觉 latent，再把这些 future latent 作为额外的 scene prior 注入到 VLA/VLM policy 的 hidden states 里，帮助动作模型在当前观测之外利用“接下来应该到哪里”的视觉信息。本 repo 把这个机制迁移到我们的 RLBench setting：用训练好的 WAN 三视角 future-video model 产生 VAE-before-decode latent，再通过 WorldPilot-style cross-attention fuser 接到 PyTorch pi0.5 上训练动作策略。

代码结构上，这里主要包含三部分：

- WAN future video latent 的 sample 对齐、导出、缓存和校验。
- WorldPilot-style `WanFutureVideoFuser`，把 `(V, C, T_lat, H_lat, W_lat)` future latent 变成可注入 pi0.5 的 hidden tokens。
- PyTorch pi0.5 latent-steering 训练和 online RLBench eval 入口，沿用 RLBench pi0.5 waypoint baseline 的 LeRobot 数据格式和 action target。
- OpenVLA-OFT experimental route，把同一个 WAN latent fuser 接到 OpenVLA-OFT continuous action-head 前的 action-token hidden states。

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

pi0.5 路线的数据格式和训练语义仍与 `main` 分支保持一致。当前 4xH100 分支额外包含一个 OpenVLA-OFT experimental route，用来评估更接近 WorldPilot hidden-state steering 的 VLA 接法。

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
  -> preserve all latent-time positions
  -> (B, V * T_lat, C * H_lat * W_lat)
  -> Linear(..., H_pi05)
  -> cross-attn into pi0.5 VLM hidden states
```

本 repo 固定保留 WAN latent-time 维度，把每个 `(view, latent_time)` 作为一个 future-scene token。

这个 repo 不使用 Wan transformer block13 hidden tokens；Latent Steering 对齐的是 **VAE-before-decode future video latent**。

### OpenVLA-OFT Route

当前分支还包含一条 OpenVLA-OFT 实验路线。它不替换 pi0.5 训练入口，而是作为第二个 VLA backend：

```text
三视角 RLBench RGB + task text + proprio
  -> OpenVLA-OFT LoRA VLA
  -> final action-token hidden states
  -> OpenVLA-OFT L1 continuous action head
  -> waypoint action chunk
```

先跑的 D1 baseline 是 **OpenVLA-OFT on RLBench, no WAN**。它使用同一套 RLBench selected10 raw RGB/lowdim/manifest，动作监督仍是 `current -> next full-task heuristic waypoint`。后续 E1 只是在同一个 OpenVLA-OFT action-token hidden state 和 L1 action head 之间插入：

```text
WAN 1-step VAE-before-decode future latent -> WanFutureVideoFuser -> residual steering
```

这条 WAN steering 路线比 pi0.5 early-prefix 版本更接近 WorldPilot，因为 OpenVLA-OFT 会显式返回 action tokens 的 final hidden states，并用 continuous action head 做 L1 regression。

本 repo 不 vendor OpenVLA-OFT，也不把 7B checkpoint 放进 git。需要的路径是：

```bash
export OPENVLA_OFT_DIR=/path/to/openvla-oft
export OPENVLA_OFT_VLA_PATH=openvla/openvla-7b
export OPENVLA_OFT_CHECKPOINT=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
```

本地写代码、跑 shape smoke、跑 RLBench raw-data dry-run 都不需要下载 OpenVLA-OFT checkpoint。真实 D1 训练会加载 `OPENVLA_OFT_VLA_PATH`，如果它是 Hugging Face repo id，就会在首次模型加载时下载到 HF cache。`OPENVLA_OFT_CHECKPOINT` 保留给官方 OpenVLA-OFT eval/reference checkpoint 使用；当前 D1 trainer 默认从 `openvla/openvla-7b` 做 LoRA fine-tune。

当前已实现 OpenVLA-OFT hidden-state steering wrapper、RLBench no-WAN D1 trainer、路径检查、wrapper shape smoke 和 RLBench raw-data smoke。OpenVLA-OFT online RLBench eval 是后续步骤，不影响现有 pi0.5 训练和 online eval 入口。

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

## Fresh 4xH100 HPC Install

这一节按一台全新的 4x H100 NVL HPC 来写。原则是：**某个部分如果已经安装好，并且对应 smoke/check 命令通过，就直接跳过该部分**。本 repo 是 sidecar repo，所以完整实验实际依赖四块东西：

```text
1. 本 repo: WorldPilot-WAN-pi0.5 sidecar scripts / fuser / training glue
2. pi0.5 baseline repo: LeRobot conversion + OpenPI config patch
3. OpenPI / pi0.5 env: PyTorch pi0.5 model, tokenizer, training code, checkpoint conversion
4. WAN + RLBench env: WAN FLF latent export/online inference, RLBench/CoppeliaSim online eval
5. OpenVLA-OFT env: only needed for the optional OpenVLA-OFT route
```

训练前只需要第 1-3 块和离线 WAN latent export 环境。真正跑 online eval 时才需要 RLBench/CoppeliaSim。

### Install 0. System Preconditions

先确认 HPC 有这些基础工具和资源：

```bash
nvidia-smi
python3 --version
git --version
git lfs version || true
```

需要：

```text
4x H100 NVL 94GB visible to CUDA
Python 3.10 or 3.11
git
git-lfs
uv
large shared/scratch storage for LeRobot data, WAN cache, checkpoints
```

如果 `uv` 没有装，可以装到用户目录：

```bash
python3 -m pip install --user uv
```

如果 HPC 用 module 管理 CUDA/Python，先在 Slurm job 或 shell 里加载对应 module。CUDA driver/runtime、NCCL、系统级 gcc 这类集群依赖如果已经由管理员配置好，就不需要在本项目里重复安装。

### Install 1. Choose Local Paths

推荐在 HPC 上只改一个主目录 `HPC_ROOT`，其他 repo、cache、checkpoint 和输出目录都从它派生。下面的 `mkdir -p` 会自动创建工作目录、cache 目录、checkpoint 目录和日志目录。

```bash
export HPC_ROOT=/scratch/$USER/rlbench_worldpilot_wan_pi05

export REPO_ROOT=${HPC_ROOT}/repos/rlbench_worldpilot_wan_pi05_latent_steering_20260628
export PI05_BASELINE_REPO=${HPC_ROOT}/repos/rlbench_pi05_waypoint_baseline_20260606
export PI05_ROOT=${HPC_ROOT}/pi05_baseline

export SELECTED1500_DATASET_ROOT=${HPC_ROOT}/data/selected1500_dataset
export WAN_BASE_MODEL=${HPC_ROOT}/models/Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers
export WAN_LORA_DIR=${HPC_ROOT}/models/wan_lora
export PYTORCH_WEIGHT_PATH=${HPC_ROOT}/checkpoints/pi05_pytorch_checkpoint

export WAN_LATENT_CACHE_ROOT=${HPC_ROOT}/cache/selected10_worldpilot_wan_latent_cache
export CHECKPOINT_BASE_DIR=${HPC_ROOT}/checkpoints/worldpilot_wan_pi05_checkpoints
export ASSETS_BASE_DIR=${HPC_ROOT}/assets/worldpilot_wan_pi05_assets
export WANDB_DIR=${HPC_ROOT}/wandb

export OPENVLA_OFT_DIR=${HPC_ROOT}/repos/openvla-oft
export OPENVLA_OFT_VLA_PATH=openvla/openvla-7b
export OPENVLA_OFT_CHECKPOINT=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
export OPENVLA_OFT_CACHE_DIR=${HPC_ROOT}/cache/openvla_oft_hf
export OPENVLA_OFT_RUN_ROOT=${HPC_ROOT}/checkpoints/openvla_oft_rlbench
export OPENVLA_OFT_STATS_PATH=${HPC_ROOT}/cache/openvla_oft_rlbench_dataset_statistics.json
export OPENVLA_OFT_NUM_IMAGES_IN_INPUT=3
export OPENVLA_OFT_PROPRIO_DIM=7

mkdir -p \
  "${HPC_ROOT}/repos" \
  "${HPC_ROOT}/data" \
  "${HPC_ROOT}/models" \
  "${HPC_ROOT}/sim" \
  "${WAN_LATENT_CACHE_ROOT}" \
  "${CHECKPOINT_BASE_DIR}" \
  "${OPENVLA_OFT_RUN_ROOT}" \
  "${ASSETS_BASE_DIR}" \
  "${OPENVLA_OFT_CACHE_DIR}" \
  "${WANDB_DIR}"
```

如果你把 raw dataset、WAN base model、WAN LoRA 或 pi0.5 checkpoint 放在别的共享盘上，只覆盖对应变量即可，不需要改其他派生路径：

```bash
export SELECTED1500_DATASET_ROOT=/shared/path/selected1500_dataset
export WAN_BASE_MODEL=/shared/path/Wan2.1-FLF2V-14B-720P-diffusers
export WAN_LORA_DIR=/shared/path/trained_wan_lora
export PYTORCH_WEIGHT_PATH=/shared/path/pi05_pytorch_checkpoint
```

注意：`mkdir -p` 只能自动创建空目录，不能自动生成数据内容。下面这些是输入资源，必须已经上传、下载或由前面步骤生成：

```text
${SELECTED1500_DATASET_ROOT}
${WAN_BASE_MODEL}
${WAN_LORA_DIR}              # 如果只做 dummy smoke，可以留空
${PYTORCH_WEIGHT_PATH}       # 如果还没有，Install 6 会从 JAX checkpoint 转换生成
${OPENVLA_OFT_DIR}           # 只在 OpenVLA-OFT route 里需要
```

### Install 2. Clone And Install This Repo

如果还没有 clone：

```bash
git clone -b hpc-4xh100-nvl \
  git@github.com:zhiyuan-gao/rlbench-worldpilot-wan-pi05-latent-steering.git \
  "${REPO_ROOT}"
```

如果已经 clone 过：

```bash
cd "${REPO_ROOT}"
git fetch origin --prune
git checkout hpc-4xh100-nvl
git pull --ff-only
```

创建本 repo 的轻量 sidecar Python 环境：

```bash
cd "${REPO_ROOT}"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install uv
pip install -e . --no-deps
pip install -r requirements.txt
```

如果 `.venv` 已经存在，可以只激活并重跑 `pip install -e . --no-deps`，这是幂等的：

```bash
cd "${REPO_ROOT}"
source .venv/bin/activate
pip install -e . --no-deps
```

### Install 3. Clone pi0.5 Baseline Repo And Install OpenPI

本实验沿用 pi0.5 waypoint baseline 的 LeRobot 数据格式和 OpenPI config。先 clone baseline repo：

```bash
git clone -b hpc-4xh100-nvl \
  git@github.com:zhiyuan-gao/rlbench-pi05-waypoint-baseline.git \
  "${PI05_BASELINE_REPO}"
```

如果已经 clone 过：

```bash
cd "${PI05_BASELINE_REPO}"
git fetch origin --prune
git checkout hpc-4xh100-nvl
git pull --ff-only
```

安装 baseline sidecar env，并让它安装/patch OpenPI：

```bash
cd "${PI05_BASELINE_REPO}"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install uv
pip install -e . --no-deps
pip install -r requirements.txt

export PI05_ROOT=${PI05_ROOT}
export SELECTED1500_DATASET_ROOT=${SELECTED1500_DATASET_ROOT}
bash scripts/install_openpi_on_hpc.sh
```

这个脚本会在 `${PI05_ROOT}/openpi` 安装 OpenPI，并 patch `pi05_rlbench_waypoint_h1` config。默认 OpenPI ref 由 baseline repo 脚本固定，适合和本 repo 对齐。

如果 `${PI05_ROOT}/openpi` 已经存在，并且你确认 OpenPI 环境已经能用，可以跳过完整安装，只重新 patch config：

```bash
cd "${PI05_BASELINE_REPO}"
source .venv/bin/activate
export OPENPI_DIR=${PI05_ROOT}/openpi
bash scripts/patch_openpi_rlbench_config.sh
```

检查 OpenPI：

```bash
cd "${PI05_ROOT}/openpi"
uv run python - <<'PY'
import openpi
import openpi.training.config as config
print("openpi:", openpi.__file__)
print(config.get_config("pi05_rlbench_waypoint_h1").name)
PY
```

### Optional Install 3b. Clone OpenVLA-OFT

这一步只对 OpenVLA-OFT route 必需。只跑 pi0.5 latent steering 时可以跳过。

如果已经有 OpenVLA-OFT checkout，并且 `bash scripts/check_openvla_oft_inputs.sh` 通过，就不用重新 clone：

```bash
cd "${REPO_ROOT}"
source .venv/bin/activate
export OPENVLA_OFT_DIR=/path/to/openvla-oft
bash scripts/check_openvla_oft_inputs.sh
```

fresh install:

```bash
cd "${REPO_ROOT}"
source .venv/bin/activate
export OPENVLA_OFT_DIR=${HPC_ROOT}/repos/openvla-oft
export OPENVLA_OFT_VLA_PATH=openvla/openvla-7b
export OPENVLA_OFT_CHECKPOINT=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
bash scripts/setup_openvla_oft_on_hpc.sh
bash scripts/check_openvla_oft_inputs.sh
```

这个步骤只安装 OpenVLA-OFT code。它不会下载 7B checkpoint。D1 训练真正读取的是 `OPENVLA_OFT_VLA_PATH`；它可以是 `openvla/openvla-7b` 这样的 Hugging Face repo id，也可以是本地 checkpoint 路径。只有真实运行 OpenVLA-OFT model load 时才会下载/读取权重。

本地或 HPC 上可以先跑不需要 checkpoint 的 shape smoke：

```bash
bash scripts/smoke_openvla_oft_shapes.sh
```

还可以先跑 RLBench raw-data smoke。它只读取 selected10 manifest、RGB path 和 `low_dim_obs.pkl`，并生成 OpenVLA-OFT 用的 action/proprio normalization stats，不会加载 7B VLA：

```bash
SPLIT=train bash scripts/smoke_openvla_oft_rlbench_data.sh
```

### Install 4. Prepare selected1500 Raw Data

把 `selected1500_dataset` 放到 HPC，并确认固定目录结构存在：

```bash
test -d "${SELECTED1500_DATASET_ROOT}/local200/rgb3_keyframes_intervals"
test -d "${SELECTED1500_DATASET_ROOT}/remote400/rgb3_keyframes_intervals"
test -d "${SELECTED1500_DATASET_ROOT}/local200/nonimage_metadata"
test -d "${SELECTED1500_DATASET_ROOT}/remote400/nonimage_metadata"
test -f "${SELECTED1500_DATASET_ROOT}/manifests/selected10_event_fullinfo_train100_val25_test25_from_train450_stratified_20260606.jsonl"
```

如果数据已经在 HPC 上并且这些检查通过，就跳过本步。

可以用 baseline repo 的检查脚本再确认一次：

```bash
cd "${PI05_BASELINE_REPO}"
source .venv/bin/activate
export PI05_ROOT=${PI05_ROOT}
export SELECTED1500_DATASET_ROOT=${SELECTED1500_DATASET_ROOT}
bash scripts/check_selected1500_dataset.sh
```

### Install 5. Convert pi0.5 LeRobot Dataset

本 repo 的 action supervision 仍然用 baseline 的 LeRobot dataset：

```text
repo_id = rlbench/selected10_pi05_waypoint_h1
target  = next full-task heuristic waypoint
```

如果这个文件已经存在，并且 smoke 通过，可以跳过 conversion：

```bash
test -f "${PI05_ROOT}/lerobot_home/rlbench/selected10_pi05_waypoint_h1/meta/info.json"
```

fresh conversion：

```bash
cd "${PI05_BASELINE_REPO}"
source .venv/bin/activate
export PI05_ROOT=${PI05_ROOT}
export SELECTED1500_DATASET_ROOT=${SELECTED1500_DATASET_ROOT}
SPLIT=train bash scripts/convert_selected10_waypoints_to_lerobot.sh
```

检查 LeRobot dataset：

```bash
cd "${PI05_BASELINE_REPO}"
source .venv/bin/activate
export PI05_ROOT=${PI05_ROOT}
bash scripts/smoke_lerobot_dataset.sh
```

计算 OpenPI normalization stats：

```bash
cd "${PI05_BASELINE_REPO}"
source .venv/bin/activate
export PI05_ROOT=${PI05_ROOT}
bash scripts/compute_norm_stats.sh
```

如果 `${PI05_ROOT}/lerobot_home/rlbench/selected10_pi05_waypoint_h1` 已经存在、`smoke_lerobot_dataset.sh` 通过、并且 OpenPI assets/norm stats 已经算过，可以跳过本步。

### Install 6. Prepare pi0.5 PyTorch Checkpoint

本 repo 训练的是 PyTorch/DDP latent-steering policy，需要 PyTorch 格式 pi0.5 checkpoint：

```bash
export PYTORCH_WEIGHT_PATH=/path/to/pi05_pytorch_checkpoint
```

如果已有 PyTorch checkpoint，确认路径存在后跳过转换：

```bash
test -e "${PYTORCH_WEIGHT_PATH}"
```

如果只有 JAX checkpoint，用 OpenPI 官方转换脚本：

```bash
cd "${PI05_ROOT}/openpi"
uv run examples/convert_jax_model_to_pytorch.py \
  --config-name pi05_rlbench_waypoint_h1 \
  --checkpoint-dir /path/to/jax/pi05_checkpoint \
  --output-path /path/to/pi05_pytorch_checkpoint
export PYTORCH_WEIGHT_PATH=/path/to/pi05_pytorch_checkpoint
```

### Install 7. Prepare WAN / Diffusers Environment

真实 WAN latent cache 和 online eval 都要求当前 Python 环境里：

```text
from diffusers import WanImageToVideoPipeline
WanImageToVideoPipeline.__call__ supports last_image
```

如果你已经有 Finetrainers/WAN 环境，并且下面检查通过，就跳过安装：

```bash
cd "${PI05_ROOT}/openpi"
uv run python - <<'PY'
import inspect
from diffusers import WanImageToVideoPipeline
print("last_image" in inspect.signature(WanImageToVideoPipeline.__call__).parameters)
PY
```

如果输出不是 `True`，需要安装或切换到支持 FLF `last_image` 的 diffusers/WAN 环境。常见做法是在 OpenPI env 里安装最新版 diffusers，或使用你训练 WAN LoRA 时的 Finetrainers env：

```bash
cd "${PI05_ROOT}/openpi"
uv pip install --python .venv/bin/python git+https://github.com/huggingface/diffusers.git
uv pip install --python .venv/bin/python accelerate transformers peft sentencepiece imageio imageio-ffmpeg
```

然后再次运行上面的 `last_image` 检查。`WAN_BASE_MODEL` 应指向 diffusers 格式的 WAN FLF base model，例如：

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-FLF2V-14B-720P-diffusers
export WAN_LORA_DIR=/path/to/trained_wan_lora
test -d "${WAN_BASE_MODEL}"
```

如果只是做 plumbing smoke，可以暂时跳过真实 WAN 安装，后续用：

```bash
WAN_LATENT_BACKEND=dummy
WAN_EXPECTED_BACKEND=dummy
```

### Install 8. Configure This Repo Paths

回到本 repo，把 HPC 真实路径写入本地 ignored file：

```bash
cd "${REPO_ROOT}"
source .venv/bin/activate
if [[ ! -f scripts/hpc_paths.sh ]]; then
  cp scripts/hpc_paths_example.sh scripts/hpc_paths.sh
fi
vim scripts/hpc_paths.sh
source scripts/hpc_paths.sh
```

至少需要确认这些变量：

```bash
echo "${REPO_ROOT}"
echo "${PI05_ROOT}"
echo "${OPENPI_DIR}"
echo "${PI05_BASELINE_REPO}"
echo "${SELECTED1500_DATASET_ROOT}"
echo "${WAN_BASE_MODEL}"
echo "${WAN_LORA_DIR}"
echo "${WAN_LATENT_CACHE_ROOT}"
echo "${PYTORCH_WEIGHT_PATH}"
```

如果 `scripts/hpc_paths.sh` 已经存在，只更新里面的路径即可，不需要重新复制模板。

总检查：

```bash
cd "${REPO_ROOT}"
source .venv/bin/activate
source scripts/hpc_paths.sh
bash scripts/check_hpc_inputs.sh
bash scripts/smoke_fuser_shapes.sh
```

### Install 9. RLBench / CoppeliaSim For Online Eval

训练和离线 WAN latent cache 不需要 CoppeliaSim。只有 online RLBench rollout eval 需要这一节。如果你只准备训练，可以先跳过。

如果 HPC 已经有 RLBench/PyRep/CoppeliaSim，并且 smoke 通过，就跳过安装：

```bash
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export RLBENCH_ROOT=/path/to/RLBench
cd "${REPO_ROOT}"
source .venv/bin/activate
source scripts/hpc_paths.sh
bash scripts/smoke_online_eval_env.sh
```

fresh install 的典型方式：

```bash
git clone https://github.com/stepjam/PyRep.git /path/to/PyRep
git clone https://github.com/stepjam/RLBench.git /path/to/RLBench
cd "${PI05_ROOT}/openpi"
uv pip install --python .venv/bin/python -e /path/to/PyRep
uv pip install --python .venv/bin/python -e /path/to/RLBench

export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export RLBENCH_ROOT=/path/to/RLBench
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${COPPELIASIM_ROOT}/lib:${LD_LIBRARY_PATH:-}"
```

再跑：

```bash
cd "${REPO_ROOT}"
source .venv/bin/activate
source scripts/hpc_paths.sh
bash scripts/smoke_online_eval_env.sh
```

## HPC Step-by-Step

这一节是把 repo 复制到 HPC 后，从路径配置到正式训练和 online RLBench eval 的完整顺序。这个 repo 是 sidecar repo：它不重新生成 pi0.5 LeRobot 数据，也不 vendor OpenPI/WAN，只在 pi0.5 baseline 数据旁边新增 WAN future-video latent cache、PyTorch latent steering 训练入口和 online rollout eval 入口。

推荐实际执行顺序：

1. 复制并填写 `scripts/hpc_paths.sh`。
2. 跑 `bash scripts/check_hpc_inputs.sh`。
3. 如果要跑 OpenVLA-OFT route，跑 `bash scripts/check_openvla_oft_inputs.sh`、`bash scripts/smoke_openvla_oft_shapes.sh` 和 `bash scripts/smoke_openvla_oft_rlbench_data.sh`。
4. 为 `train`/`val` 构建 sample index。
5. 跑 dummy cache smoke 和 train dry-run。
6. 导出真实 `wan-diffusers` WAN latent cache，并 validate `train`/`val`。
7. 准备或转换 pi0.5 PyTorch checkpoint。
8. 训练 PyTorch/DDP latent-steering policy。
9. 直接跑 online RLBench rollout eval。

如果只跑 OpenVLA-OFT D1 no-WAN baseline，可以跳过第 5-7 步里的 WAN cache 和 pi0.5 checkpoint，直接使用 `scripts/train_openvla_oft_rlbench.sh` 训练 OpenVLA-OFT。只有后续 OpenVLA-OFT + WAN steering 版本才需要第 6 步的 WAN latent cache。

### 1. Configure Required Paths

HPC 上不要改源码里的默认路径。推荐复制路径模板，然后只改这个本地文件：

```bash
cd ${REPO_ROOT}
if [[ ! -f scripts/hpc_paths.sh ]]; then
  cp scripts/hpc_paths_example.sh scripts/hpc_paths.sh
fi
vim scripts/hpc_paths.sh
source scripts/hpc_paths.sh
```

`scripts/hpc_paths.sh` 已经被 `.gitignore` 忽略，可以写 HPC 的真实路径。所有训练、cache 和 online eval 脚本都会再次 source `scripts/setup_env.sh`；`setup_env.sh` 会保留你已经 export 的变量，所以 `scripts/hpc_paths.sh` 或 Slurm 里的 `export` 会覆盖本机默认路径。

如果不想建 `scripts/hpc_paths.sh`，也可以直接在 Slurm 脚本里写同样的 `export`。至少需要这些路径：

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

# Optional OpenVLA-OFT route.
export OPENVLA_OFT_DIR=/path/to/openvla-oft
export OPENVLA_OFT_VLA_PATH=openvla/openvla-7b
export OPENVLA_OFT_CACHE_DIR=/scratch/path/openvla_oft_hf
export OPENVLA_OFT_RUN_ROOT=/scratch/path/openvla_oft_rlbench_checkpoints
export OPENVLA_OFT_STATS_PATH=/scratch/path/openvla_oft_rlbench_dataset_statistics.json
export OPENVLA_OFT_NUM_IMAGES_IN_INPUT=3
export OPENVLA_OFT_PROPRIO_DIM=7
```

其中 `HF_LEROBOT_HOME` 必须能找到 pi0.5 baseline 已经转换好的 LeRobot 数据：

```text
${HF_LEROBOT_HOME}/rlbench/selected10_pi05_waypoint_h1/meta/info.json
```

如果这个文件不存在，需要先回到 `rlbench_pi05_waypoint_baseline_20260606` 按 baseline 流程生成 LeRobot dataset。

改完路径后先跑：

```bash
source scripts/hpc_paths.sh
bash scripts/check_hpc_inputs.sh
```

它会检查 OpenPI、baseline repo、selected1500 raw RGB/lowdim、manifest、WAN base model 和 LeRobot dataset 是否能找到。

### 2. Configure WAN Latent Cache Paths

WAN latent cache 是本 repo 新增的数据。建议放在 scratch 或大容量共享盘：

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-FLF2V-14B-720P-diffusers
export WAN_LORA_DIR=/path/to/trained_wan_lora
export WAN_LATENT_CACHE_ROOT=/scratch/path/selected10_worldpilot_wan_latent_cache
export WAN_NUM_INFERENCE_STEPS=1
export WAN_OUTPUT_LAYOUT=bcthw
export WAN_EXPECTED_BACKEND=wan-diffusers
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

`WAN_OUTPUT_LAYOUT` 是 WAN pipeline 返回 latent 的布局，默认 `bcthw`，即 `[B,C,T,H,W]`。如果 patched pipeline 返回 `[B,T,C,H,W]`，设成 `btchw`。代码会校验 `C=16` 和 expected `T_lat`，layout 写错会直接报错，不会静默交换 channel/time。

`WAN_EXPECTED_BACKEND` 是训练和 cache validate 对 cache 的 backend 校验。正式训练默认应该是 `wan-diffusers`，dummy smoke 时临时设成 `dummy`。

`WAN_LATENT_SHAPE` 是训练和 validate 对 cache 的 shape 校验，默认 `3,16,6,32,32`，对应 `[V,C,T_lat,H_lat,W_lat]`。Online eval 不读取离线 cache，但会用同一个 shape 初始化 pi0.5 WAN fuser。

如果之前已经用其他 denoise step 数导出过 cache，不要和当前 1-step 实验混用；建议换一个新的 `WAN_LATENT_CACHE_ROOT` 或用 `--overwrite` 重新导出。

每个 `.pt` 文件里主要保存：

```text
future_video_latents: Tensor[V, C, T_lat, H_lat, W_lat]
latent_layout:        vcthw
view_names:           ["front", "left_shoulder", "right_shoulder"]
```

当前 exporter 默认 `WAN_NUM_FRAMES=21`、`height=256`、`view_width=256`、三视角 hstack。Wan VAE 的 temporal scale 是 4、spatial scale 是 8，所以默认 per-view latent shape 是：

```text
future_video_latents: Tensor[3, 16, 6, 32, 32]
T_lat = (21 - 1) // 4 + 1
H_lat = 256 // 8
W_lat = 256 // 8
```

如果之后改 `--num-frames`、`--height` 或 `--view-width`，dummy cache、validate、train 和 online eval 的 `WAN_LATENT_SHAPE` 也要跟着改。

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

### 4b. Optional OpenVLA-OFT D1 No-WAN Baseline

这一步训练原版 OpenVLA-OFT RLBench baseline，不使用 WAN latent，也不需要 pi0.5 PyTorch checkpoint。它和 pi0.5 baseline 使用同一个 selected10 raw 数据和 waypoint target：

```text
input:  front + left_shoulder + right_shoulder RGB, task text, current absolute rotvec7 proprio
target: current -> next full-task heuristic waypoint absolute rotvec7 action
chunk:  OpenVLA-OFT 8-step L1 chunk, 当前实现把同一个 waypoint target repeat 成 8 个 action slots
```

先确认 raw data 和 normalization stats：

```bash
SPLIT=train bash scripts/smoke_openvla_oft_rlbench_data.sh
```

真实训练会首次下载/加载 `OPENVLA_OFT_VLA_PATH`，默认是 `openvla/openvla-7b`：

```bash
export EXP_NAME=rlbench_openvla_oft_waypoint_no_wan
export OPENVLA_OFT_RUN_ROOT=/scratch/path/openvla_oft_rlbench_checkpoints
export OPENVLA_OFT_CACHE_DIR=/scratch/path/openvla_oft_hf

NPROC_PER_NODE=4 SPLIT=train \
bash scripts/train_openvla_oft_rlbench.sh \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --max-steps 100000 \
  --save-interval 5000
```

默认保存内容是 LoRA adapter、OpenVLA-OFT L1 `action_head.pt`、`proprio_projector.pt`、optimizer state 和 `dataset_statistics.json`；不会在每个 checkpoint 自动 merge/save 7B 全量模型。

### 5. Dummy Cache Smoke Test

先不要跑真实 WAN，先用 dummy latent 确认 sample index、cache 读取和 OpenPI dataloader 能连起来：

```bash
WAN_LATENT_BACKEND=dummy WAN_EXPECTED_BACKEND=dummy SPLIT=train \
bash scripts/export_wan_latent_cache.sh --max-samples 16 --overwrite

WAN_EXPECTED_BACKEND=dummy SPLIT=train \
bash scripts/validate_wan_latent_cache.sh --max-samples 16
```

`--max-samples` 只限制本次导出的 cache 数量，不会把 `sample_index_train.jsonl` 截短。`validate_wan_latent_cache.sh` 会同时检查 cache 里的 `metadata.num_inference_steps` 是否等于当前 `WAN_NUM_INFERENCE_STEPS`，并检查 latent shape 是否等于 `WAN_LATENT_SHAPE`。

然后做训练 dataloader dry-run：

```bash
export PYTORCH_WEIGHT_PATH=/path/to/pi05_pytorch_checkpoint

SPLIT=train \
WAN_NUM_INFERENCE_STEPS=1 \
WAN_EXPECTED_BACKEND=dummy \
NPROC_PER_NODE=4 \
bash scripts/train_worldpilot_wan_pi05_torch.sh \
  --dry-run \
  --allow-missing-latents \
  --batch-size 4 \
  --no-wandb-enabled
```

`--dry-run` 不启动训练，也不加载 pi0.5 base weights；它用于检查 transformed LeRobot dataset、sample index、latent cache batch 和 split/step 配置。fuser shape 单测仍然用 `bash scripts/smoke_fuser_shapes.sh`。

### 6. Export Real WAN Latent Cache

dummy smoke test 通过后，导出真实 WAN VAE-before-decode future video latent：

```bash
WAN_LATENT_BACKEND=wan-diffusers WAN_EXPECTED_BACKEND=wan-diffusers WAN_NUM_INFERENCE_STEPS=1 SPLIT=train \
bash scripts/export_wan_latent_cache.sh --resume

WAN_LATENT_BACKEND=wan-diffusers WAN_EXPECTED_BACKEND=wan-diffusers WAN_NUM_INFERENCE_STEPS=1 SPLIT=val \
bash scripts/export_wan_latent_cache.sh --resume
```

数据量大时可以用 shard 并行。例如 8 个 Slurm jobs：

```bash
WAN_LATENT_BACKEND=wan-diffusers WAN_EXPECTED_BACKEND=wan-diffusers WAN_NUM_INFERENCE_STEPS=1 SPLIT=train \
bash scripts/export_wan_latent_cache.sh \
  --num-shards 8 \
  --shard-index 0 \
  --resume
```

其他 jobs 分别使用 `--shard-index 1` 到 `--shard-index 7`。

导出后检查 coverage 和 shape：

```bash
WAN_EXPECTED_BACKEND=wan-diffusers SPLIT=train bash scripts/validate_wan_latent_cache.sh
WAN_EXPECTED_BACKEND=wan-diffusers SPLIT=val bash scripts/validate_wan_latent_cache.sh
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

训练结束后直接进入 Step 9；本实验的论文结果应使用 online RLBench rollout eval。

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
WAN_OUTPUT_LAYOUT         bcthw for [B,C,T,H,W], btchw for [B,T,C,H,W]
WAN_LATENT_SHAPE          expected [V,C,T_lat,H_lat,W_lat], default 3,16,6,32,32
WAN_EXPECTED_BACKEND      wan-diffusers for real training, dummy for dummy smoke
PYTORCH_WEIGHT_PATH       pi0.5 PyTorch checkpoint
OPENVLA_OFT_DIR           optional OpenVLA-OFT checkout for OpenVLA-OFT route
OPENVLA_OFT_VLA_PATH      OpenVLA base VLA for D1 training, default openvla/openvla-7b
OPENVLA_OFT_CHECKPOINT    optional official eval/reference checkpoint, not used by D1 trainer
OPENVLA_OFT_CACHE_DIR     optional HF cache dir for OpenVLA-OFT downloads
OPENVLA_OFT_RUN_ROOT      output directory for OpenVLA-OFT RLBench checkpoints
OPENVLA_OFT_STATS_PATH    action/proprio normalization stats JSON
CHECKPOINT_BASE_DIR       output directory for this experiment
```

### 9. Run Online RLBench Rollout Eval

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

默认 smoke 使用当前 exporter 默认 WAN latent size：`[B, 3, 16, 6, 32, 32]`，对应 `WAN_NUM_FRAMES=21,height=256,view_width=256`。如果改了 WAN 输入分辨率或帧数，需要给 smoke 脚本显式传 `--latent-steps/--height/--width`。

## Quick Reference: WAN Latent Cache

先构建与 pi0.5 LeRobot samples 对齐的 sample index。默认 `WAN_LATENT_GOAL_MODE=event_end`，所以 action target 仍是 next waypoint，WAN latent goal 是当前 event/subgoal end：

```bash
SPLIT=train \
bash scripts/build_sample_index.sh
```

快速 pipeline smoke 可以先导出 dummy latent cache：

```bash
WAN_LATENT_BACKEND=dummy \
WAN_EXPECTED_BACKEND=dummy \
SPLIT=train \
bash scripts/export_wan_latent_cache.sh --max-samples 16 --overwrite
```

真实 WAN VAE-before-decode latent cache：

```bash
WAN_LATENT_BACKEND=wan-diffusers \
WAN_BASE_MODEL=/raid/home/than/zhiyuan/finetrainers/pretrained_models/Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers \
WAN_LORA_DIR=/path/to/wan_lora \
WAN_NUM_INFERENCE_STEPS=1 \
WAN_OUTPUT_LAYOUT=bcthw \
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
WAN_EXPECTED_BACKEND=wan-diffusers \
SPLIT=train \
bash scripts/validate_wan_latent_cache.sh
```

## Quick Reference: OpenVLA-OFT Route

OpenVLA-OFT route 是 optional，不影响 pi0.5 训练脚本。它的目标接法是：

```text
D1 no-WAN:
  RLBench RGB/text/proprio -> OpenVLA-OFT -> final action-token hidden states
  -> OpenVLA-OFT L1 continuous action head

E1 WAN steering:
  final action-token hidden states + WAN future latent residual
  -> OpenVLA-OFT L1 continuous action head
```

只检查本 repo 的 wrapper shape，不需要 OpenVLA-OFT repo，也不需要 checkpoint：

```bash
bash scripts/smoke_openvla_oft_shapes.sh
```

检查 OpenVLA-OFT external checkout 和 checkpoint 配置：

```bash
export OPENVLA_OFT_DIR=/path/to/openvla-oft
export OPENVLA_OFT_VLA_PATH=openvla/openvla-7b
bash scripts/check_openvla_oft_inputs.sh
```

检查 RLBench raw data/stats，不下载 7B 权重：

```bash
SPLIT=train bash scripts/smoke_openvla_oft_rlbench_data.sh
```

训练 D1 no-WAN baseline：

```bash
EXP_NAME=rlbench_openvla_oft_waypoint_no_wan \
NPROC_PER_NODE=4 SPLIT=train \
bash scripts/train_openvla_oft_rlbench.sh \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --max-steps 100000 \
  --save-interval 5000
```

如果 `OPENVLA_OFT_VLA_PATH` 是 Hugging Face repo id，这个 check 和 raw-data smoke 不会下载权重；只有真实 OpenVLA-OFT model load 时才会下载到 HF cache。如果 HPC 已经提前下载好了 checkpoint，也可以把 `OPENVLA_OFT_VLA_PATH` 指向本地目录。

## Quick Reference: Training And Online Eval

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

训练完成后直接跑 Online RLBench rollout eval：

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
- online RLBench rollout eval entry with event/subgoal goal scheduling and per-step WAN latent refresh

仍需要在 HPC 真机上验证：

- `wan-diffusers` backend 返回 latent 的实际 shape 是否和 diffusers 版本一致
- 真实 WAN LoRA 路径和显存配置
- online RLBench rollout 的 CoppeliaSim/RLBench 运行、WAN online backend 显存配置和成功率
