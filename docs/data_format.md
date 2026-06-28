# Data Format

本 repo 的监督数据格式沿用 RLBench pi0.5 waypoint baseline。

## Base LeRobot Dataset

默认数据：

```text
HF/LEROBOT repo id: rlbench/selected10_pi05_waypoint_h1
```

每个训练 sample 使用：

```text
front_image            uint8   [256, 256, 3]
left_shoulder_image    uint8   [256, 256, 3]
right_shoulder_image   uint8   [256, 256, 3]
state                  float32 [7]
actions                float32 [7]
task                   string
```

动作格式：

```text
absolute_rotvec7 = x, y, z, rx, ry, rz, gripper_open
```

动作监督 target 仍然是下一个 full-task heuristic waypoint，`action_horizon=1`。WAN steering latent 的 goal 不是这个 next waypoint，而是当前 sample 所属 event/subgoal 的 end frame。WAN latent 默认取一步 denoising 后、VAE decode 前的 future latent。

## Manifest

默认 action/waypoint manifest 来自 pi0.5 baseline repo：

```text
/raid/home/than/zhiyuan/corl2026/rlbench_pi05_waypoint_baseline_20260606/manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl
```

这些字段用于把 LeRobot sample 和原始 RLBench episode/frame/action target 对齐：

```text
split
source_bundle
rgb_episode_relpath
task
variation
episode
task_instruction
full_task_heuristic_waypoints
num_frames
```

event/subgoal manifest 默认来自 selected1500：

```text
/raid/home/than/zhiyuan/selected1500_dataset/manifests/selected10_event_fullinfo_train100_val25_test25_from_train450_stratified_20260606.jsonl
```

它提供 `event_keyframes_adjusted` / `event_heuristic_waypoints_by_event`，用于把每个 current frame 映射到当前 event/subgoal end：

```text
current frame                -> action_target_waypoint_frame = next full-task waypoint
current frame within event i -> latent_goal_frame = event_keyframes_adjusted[i + 1]
WAN denoise depth            -> num_inference_steps = 1
```

## Raw RGB / Low-Dim Roots

默认 raw 数据根目录和当前项目里的 selected1500 路径一致：

```text
SELECTED1500_DATASET_ROOT=/raid/home/than/zhiyuan/selected1500_dataset
RGB_ROOT_200=${SELECTED1500_DATASET_ROOT}/local200/rgb3_keyframes_intervals
RGB_ROOT_400=${SELECTED1500_DATASET_ROOT}/remote400/rgb3_keyframes_intervals
LOWDIM_ROOT_200=${SELECTED1500_DATASET_ROOT}/local200/nonimage_metadata
LOWDIM_ROOT_400=${SELECTED1500_DATASET_ROOT}/remote400/nonimage_metadata
```

`source_bundle=all200` 使用 `RGB_ROOT_200/LOWDIM_ROOT_200`，`source_bundle=all400` 使用 `RGB_ROOT_400/LOWDIM_ROOT_400`。WAN latent export 当前只需要 RGB roots；LeRobot conversion 和 online eval 会使用 low-dim roots。

Online RLBench eval 不读取离线 WAN latent cache。它使用 low-dim episode demo 做 `reset_to_demo`，用 raw RGB roots 读取当前 event/subgoal end 的 oracle goal image，然后在每个控制步用 live RLBench RGB 重新跑 WAN latent provider。

## WAN Latent Cache

本 repo 额外需要 WAN future video latent cache。推荐每个 sample 一个文件，或者每个 episode 一个 shard；无论存储粒度如何，读取出来的单个 sample 应能返回：

```text
future_video_latents: Tensor[V, C, T_lat, H_lat, W_lat]
view_names:           list[str]
task:                 str
variation:            int
episode:              int
source_bundle:        str
frame_index:          int
target_waypoint_frame:int
action_target_waypoint_frame:int
latent_goal_frame:    int
event_idx:            int
event_end_frame:      int
instruction:          str
latent_layout:        "vcthw"
metadata.num_inference_steps: 1
```

如果实际提取脚本更容易保存成 `(V, T_lat, C, H_lat, W_lat)`，也可以，但 metadata 必须写清 `latent_layout="vtchw"`，加载时再统一转换。

## Branch Difference

`main` 和 `hpc-4xh100-nvl` 不能改变数据格式。它们只区别：

- 默认 GPU 数。
- 默认 HPC 显存建议。
- training launcher 的 `NPROC_PER_NODE` 默认值。
- README 中的机器描述。
