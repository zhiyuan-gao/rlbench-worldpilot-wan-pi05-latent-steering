# Block12 per-event evaluation v1

该协议用于正式评估 GFPI-Frozen Block12，不改变模型结构或训练权重。

## 固定行为

1. 使用 `val` split 中预先确定的前 25 个 episodes；正式 seeds 为 `0, 1, 2`。
2. π0.5 action noise 每个控制步使用一个由 `eval_seed + task + variation + episode + step` 稳定生成的新 seed。
3. WAN noise 在同一个 event 内固定，切换 event 后更新；seed 由 `eval_seed + task + variation + episode + event` 稳定生成。
4. event 按末端位姿到达或 8-step 上限切换。
5. 到达最后一个 event 后，如果 RLBench 尚未成功，则锁定最后一个 goal 并继续闭环执行。
6. rollout 只在 RLBench 成功、invalid action 或 30-step 上限时结束。

## 通用入口

- evaluator：`src/rlbench_worldpilot_wan_pi05/eval_online_rlbench_rpc_block12_per_event_v1.py`
- 双 GPU RPC launcher：`scripts/run_block12_per_event_v1_rpc.sh`
- 单元测试：`tests/test_eval_online_rlbench_rpc_block12_per_event_v1.py`

launcher 通过环境变量接收 checkpoint、WAN、RLBench、PyRep、CoppeliaSim、数据集和 GPU 配置；不包含服务器专用输出路径。
