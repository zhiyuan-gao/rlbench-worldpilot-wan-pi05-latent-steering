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

训练 target 仍然是下一个 full-task heuristic waypoint，`action_horizon=1`。

## Manifest

默认 manifest 来自 pi0.5 baseline repo：

```text
/raid/home/than/zhiyuan/corl2026/rlbench_pi05_waypoint_baseline_20260606/manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl
```

这些字段用于把 LeRobot sample 和原始 RLBench episode/frame 对齐：

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
instruction:          str
latent_layout:        "vcthw"
```

如果实际提取脚本更容易保存成 `(V, T_lat, C, H_lat, W_lat)`，也可以，但 metadata 必须写清 `latent_layout="vtchw"`，加载时再统一转换。

## Branch Difference

`main` 和 `hpc-4xh100-nvl` 不能改变数据格式。它们只区别：

- 默认 GPU 数。
- 默认 HPC 显存建议。
- training launcher 的 `NPROC_PER_NODE` 默认值。
- README 中的机器描述。

