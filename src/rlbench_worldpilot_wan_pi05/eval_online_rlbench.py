from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import traceback
from typing import Any

import numpy as np

from .online_policy import WanPi05OnlinePolicy
from .online_wan import DummyWanLatentProvider, WanDiffusersOnlineProvider, hstack_views, parse_latent_shape
from .rlbench_online_utils import (
    action7_to_rlbench_action9,
    install_rlbench_root,
    load_demo,
    make_action_mode,
    make_obs_config,
    obs_rgb_images,
    policy_obs_from_rlbench,
    pose_event_reached,
    read_goal_images_from_episode,
    resolve_episode_dir,
    task_class_from_name,
)
from .sample_index import event_boundaries, read_jsonl, row_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online RLBench rollout eval for WorldPilot-style WAN pi0.5.")
    parser.add_argument("config_name", nargs="?", default=os.environ.get("CONFIG_NAME", "pi05_rlbench_waypoint_h1"))
    parser.add_argument("--exp-name", default=os.environ.get("EXP_NAME", "selected10_worldpilot_wan_pi05_torch"))
    parser.add_argument("--eval-checkpoint", default=os.environ.get("EVAL_CHECKPOINT"))
    parser.add_argument("--checkpoint-base-dir", default=os.environ.get("CHECKPOINT_BASE_DIR", "./checkpoints"))
    parser.add_argument("--assets-base-dir", default=os.environ.get("ASSETS_BASE_DIR", "./assets"))
    parser.add_argument("--out", type=Path, default=os.environ.get("ONLINE_EVAL_OUT"))

    parser.add_argument("--manifest-path", type=Path, default=os.environ.get("MANIFEST_PATH"), required=os.environ.get("MANIFEST_PATH") is None)
    parser.add_argument("--event-manifest-path", type=Path, default=os.environ.get("EVENT_MANIFEST_PATH"), required=os.environ.get("EVENT_MANIFEST_PATH") is None)
    parser.add_argument("--split", default=os.environ.get("SPLIT", "val"), choices=("train", "val", "test", "all"))
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=25)
    parser.add_argument("--selection", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--rlbench-root", default=os.environ.get("RLBENCH_ROOT", "/raid/home/than/zhiyuan/RLBench"))
    parser.add_argument("--lowdim-root-200", type=Path, default=os.environ.get("LOWDIM_ROOT_200"))
    parser.add_argument("--lowdim-root-400", type=Path, default=os.environ.get("LOWDIM_ROOT_400"))
    parser.add_argument("--rgb-root-200", type=Path, default=os.environ.get("RGB_ROOT_200"))
    parser.add_argument("--rgb-root-400", type=Path, default=os.environ.get("RGB_ROOT_400"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--policy-device", default=os.environ.get("POLICY_DEVICE", "cuda:0"))
    parser.add_argument("--pytorch-training-precision", choices=("bfloat16", "float32"), default=os.environ.get("PYTORCH_TRAINING_PRECISION"))
    parser.add_argument("--wan-latent-shape", default=os.environ.get("WAN_LATENT_SHAPE", "3,16,5,28,28"))
    parser.add_argument("--wan-latent-time-mode", choices=("all", "last", "mean"), default=os.environ.get("WAN_LATENT_TIME_MODE", "all"))
    parser.add_argument("--wan-fuser-num-heads", type=int, default=int(os.environ.get("WAN_FUSER_NUM_HEADS", "8")))
    parser.add_argument("--wan-fuser-dropout", type=float, default=float(os.environ.get("WAN_FUSER_DROPOUT", "0.0")))
    parser.add_argument("--action-num-steps", type=int, default=10)

    parser.add_argument("--wan-backend", choices=("dummy", "wan-diffusers"), default=os.environ.get("WAN_LATENT_BACKEND", "dummy"))
    parser.add_argument("--wan-base-model", type=Path, default=os.environ.get("WAN_BASE_MODEL"))
    parser.add_argument("--wan-lora-dir", type=Path, default=os.environ.get("WAN_LORA_DIR") or None)
    parser.add_argument("--wan-height", type=int, default=256)
    parser.add_argument("--wan-view-width", type=int, default=256)
    parser.add_argument("--wan-num-frames", type=int, default=21)
    parser.add_argument("--wan-num-inference-steps", type=int, default=int(os.environ.get("WAN_NUM_INFERENCE_STEPS", "1")))
    parser.add_argument("--wan-guidance-scale", type=float, default=1.0)
    parser.add_argument("--wan-lora-scale", type=float, default=1.0)
    parser.add_argument("--wan-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--wan-device-map", default=os.environ.get("WAN_DEVICE_MAP", "balanced"))
    parser.add_argument("--wan-text-source", choices=("task", "event_description", "event_task_instruction"), default="task")

    parser.add_argument("--event-switch-mode", choices=("pose_or_steps", "fixed_steps", "steps_only", "never"), default="pose_or_steps")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-steps-per-event", type=int, default=8)
    parser.add_argument("--fixed-steps-per-event", type=int, default=None)
    parser.add_argument("--event-goal-pos-threshold", type=float, default=0.04)
    parser.add_argument("--event-goal-rot-threshold", type=float, default=0.5)
    parser.add_argument("--event-goal-gripper-threshold", type=float, default=0.5)
    parser.add_argument("--state-gripper-threshold", type=float, default=0.95)
    parser.add_argument("--gripper-exec-open-threshold", type=float, default=0.95)
    parser.add_argument("--ignore-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arm-success-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-invalid", action="store_true")

    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--record-failures-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=2)
    return parser.parse_args()


def parse_tasks(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    tasks: set[str] = set()
    for value in values:
        tasks.update(item for item in str(value).replace(",", " ").split() if item)
    return tasks or None


def root_for_source(row: dict[str, Any], root_200: Path, root_400: Path) -> Path:
    source = str(row.get("source_bundle", "all200"))
    if source in {"all200", "local200"}:
        return Path(root_200)
    if source in {"all400", "remote400"}:
        return Path(root_400)
    raise KeyError(f"Unsupported source_bundle={source!r}")


def merged_eval_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    action_lookup = {row_key(row): row for row in read_jsonl(args.manifest_path)}
    task_filter = parse_tasks(args.task)
    rows = []
    per_task = defaultdict(int)
    for event_row in read_jsonl(args.event_manifest_path):
        if args.split != "all" and str(event_row.get("split")) != args.split:
            continue
        task = str(event_row.get("task"))
        if task_filter is not None and task not in task_filter:
            continue
        if args.max_episodes is not None and len(rows) >= args.max_episodes:
            break
        if args.max_episodes_per_task is not None and per_task[task] >= args.max_episodes_per_task:
            continue
        action_row = action_lookup.get(row_key(event_row), {})
        row = dict(event_row)
        for key in ("task_instruction", "rgb_episode_relpath", "full_task_heuristic_waypoints"):
            if not row.get(key) and action_row.get(key):
                row[key] = action_row[key]
        if not row.get("task_instruction"):
            row["task_instruction"] = task.replace("_", " ")
        rows.append(row)
        per_task[task] += 1
    if args.selection == "random":
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if not rows:
        raise RuntimeError("No eval rows matched the requested filters.")
    return rows


def wan_prompt_for_event(row: dict[str, Any], event_idx: int, source: str) -> str:
    if source == "event_description":
        descriptions = row.get("event_descriptions")
        if isinstance(descriptions, list) and 0 <= event_idx < len(descriptions):
            return str(descriptions[event_idx])
    if source == "event_task_instruction":
        instructions = row.get("event_task_instructions")
        if isinstance(instructions, list) and 0 <= event_idx < len(instructions):
            return str(instructions[event_idx])
    return str(row.get("task_instruction") or row.get("task") or "")


def event_fixed_step_budget(row: dict[str, Any], event_idx: int, args: argparse.Namespace) -> int:
    if args.fixed_steps_per_event is not None:
        return int(args.fixed_steps_per_event)
    counts = row.get("event_waypoint_counts")
    if isinstance(counts, list) and 0 <= event_idx < len(counts):
        return max(1, int(counts[event_idx]))
    return int(args.max_steps_per_event)


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> str | None:
    if not frames:
        return None
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path.as_posix(), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path.as_posix()


def run_episode(
    *,
    row_idx: int,
    row: dict[str, Any],
    task_env,
    policy: WanPi05OnlinePolicy,
    wan_provider,
    args: argparse.Namespace,
) -> dict[str, Any]:
    lowdim_root = root_for_source(row, args.lowdim_root_200, args.lowdim_root_400)
    rgb_root = root_for_source(row, args.rgb_root_200, args.rgb_root_400)
    lowdim_episode_dir = resolve_episode_dir(row, lowdim_root)
    rgb_episode_dir = resolve_episode_dir(row, rgb_root)

    demo = load_demo(lowdim_episode_dir)
    observations = getattr(demo, "_observations", demo)
    boundaries = event_boundaries(row, int(row.get("num_frames", len(observations))))
    _, obs = task_env.reset_to_demo(demo)

    frames: list[np.ndarray] = []
    if args.record_video:
        frames.append(np.asarray(hstack_views(obs_rgb_images(obs), height=args.image_size, view_width=args.image_size)))

    event_idx = 0
    event_step_idx = 0
    success = False
    terminal = False
    invalid_actions = 0
    step_results = []
    goal_images = None
    goal_frame = None
    goal_obs = None

    for step_idx in range(int(args.max_steps)):
        if event_idx >= len(boundaries) - 1:
            break
        event_end = int(boundaries[event_idx + 1])
        if goal_frame != event_end:
            goal_frame = event_end
            goal_images = read_goal_images_from_episode(rgb_episode_dir, goal_frame)
            goal_obs = observations[goal_frame]

        assert goal_images is not None and goal_obs is not None
        current_images = obs_rgb_images(obs)
        prompt = str(row.get("task_instruction") or row.get("task") or "")
        wan_prompt = wan_prompt_for_event(row, event_idx, args.wan_text_source)
        wan_latents = wan_provider(
            current_images,
            goal_images,
            prompt=wan_prompt,
            seed=int(args.seed) + row_idx * 1000 + step_idx,
        )
        action7 = policy.infer_action7(
            policy_obs_from_rlbench(obs, prompt=prompt, gripper_threshold=args.state_gripper_threshold),
            wan_latents,
        )
        action9 = action7_to_rlbench_action9(action7, ignore_collision=args.ignore_collision)

        result = {
            "step_idx": int(step_idx),
            "event_idx": int(event_idx),
            "event_step_idx": int(event_step_idx),
            "event_end_frame": int(event_end),
            "wan_prompt": wan_prompt,
            "wan_latents_shape": list(wan_latents.shape),
            "action7": np.asarray(action7, dtype=float).tolist(),
            "action9": action9.tolist(),
            "invalid_action": False,
        }
        try:
            obs, reward, terminal = task_env.step(action9)
        except Exception as exc:
            invalid_actions += 1
            result.update(
                {
                    "invalid_action": True,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=6),
                }
            )
            step_results.append(result)
            if not args.continue_on_invalid:
                break
            continue

        success = bool(float(reward) > 0.0 or terminal)
        event_step_idx += 1
        reached, reach_metrics = pose_event_reached(
            obs,
            goal_obs,
            pos_threshold=args.event_goal_pos_threshold,
            rot_threshold=args.event_goal_rot_threshold,
            gripper_threshold=args.event_goal_gripper_threshold,
        )
        switch_reason = None
        fixed_budget = event_fixed_step_budget(row, event_idx, args)
        if args.event_switch_mode == "pose_or_steps":
            if reached:
                switch_reason = "pose_reached"
            elif event_step_idx >= args.max_steps_per_event:
                switch_reason = "max_steps_per_event"
        elif args.event_switch_mode == "fixed_steps":
            if event_step_idx >= fixed_budget:
                switch_reason = "fixed_steps"
        elif args.event_switch_mode == "steps_only":
            if event_step_idx >= args.max_steps_per_event:
                switch_reason = "max_steps_per_event"

        result.update(
            {
                "reward": float(reward),
                "terminal": bool(terminal),
                "success_after_step": bool(success),
                "event_reached_by_pose": bool(reached),
                "event_switch_reason": switch_reason,
                **reach_metrics,
            }
        )
        step_results.append(result)

        if args.record_video:
            frames.append(np.asarray(hstack_views(obs_rgb_images(obs), height=args.image_size, view_width=args.image_size)))
        if success:
            break
        if switch_reason is not None:
            event_idx += 1
            event_step_idx = 0

    video_path = None
    if args.record_video and frames and (not args.record_failures_only or not success):
        video_path = write_video(
            args.out.parent / "videos" / f"{row_idx:05d}__{row['task']}__{row['variation']}__{row['episode']}.mp4",
            frames,
            fps=args.video_fps,
        )

    return {
        "success": bool(success),
        "terminal": bool(terminal),
        "invalid_actions": int(invalid_actions),
        "executed_steps": len(step_results),
        "event_switch_mode": args.event_switch_mode,
        "event_boundaries": [int(x) for x in boundaries],
        "video_path": video_path,
        "step_results": step_results,
    }


def make_wan_provider(args: argparse.Namespace):
    shape = parse_latent_shape(args.wan_latent_shape)
    if args.wan_backend == "dummy":
        return DummyWanLatentProvider(shape)
    if args.wan_base_model is None:
        raise ValueError("--wan-base-model is required for --wan-backend wan-diffusers")
    return WanDiffusersOnlineProvider(
        base_model=args.wan_base_model,
        lora_dir=args.wan_lora_dir,
        height=args.wan_height,
        view_width=args.wan_view_width,
        num_frames=args.wan_num_frames,
        num_inference_steps=args.wan_num_inference_steps,
        guidance_scale=args.wan_guidance_scale,
        lora_scale=args.wan_lora_scale,
        dtype=args.wan_dtype,
        device_map=args.wan_device_map,
    )


def main() -> None:
    args = parse_args()
    if args.out is None:
        args.out = Path("online_eval") / f"{args.exp_name}_{args.split}.jsonl"
    args.out = Path(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    install_rlbench_root(args.rlbench_root)
    from rlbench.environment import Environment

    rows = merged_eval_rows(args)
    policy = WanPi05OnlinePolicy(
        config_name=args.config_name,
        exp_name=args.exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        assets_base_dir=args.assets_base_dir,
        eval_checkpoint=args.eval_checkpoint,
        wan_latent_shape=parse_latent_shape(args.wan_latent_shape),
        device=args.policy_device,
        precision=args.pytorch_training_precision,
        wan_time_mode=args.wan_latent_time_mode,
        wan_num_heads=args.wan_fuser_num_heads,
        wan_dropout=args.wan_fuser_dropout,
        action_num_steps=args.action_num_steps,
    )
    wan_provider = make_wan_provider(args)

    action_mode = make_action_mode(
        gripper_open_threshold=args.gripper_exec_open_threshold,
        stop_on_success=args.arm_success_stop,
    )
    env = Environment(
        action_mode,
        dataset_root=str(args.lowdim_root_200 or ""),
        obs_config=make_obs_config(image_size=args.image_size),
        headless=bool(args.headless),
        static_positions=False,
    )
    env.launch()

    counts = defaultdict(lambda: {"n": 0, "success": 0, "fail": 0, "invalid_actions": 0, "errors": 0})
    current_task = None
    task_env = None
    try:
        with args.out.open("w", encoding="utf-8") as f:
            for row_idx, row in enumerate(rows):
                task = str(row["task"])
                if task != current_task:
                    task_env = env.get_task(task_class_from_name(task))
                    current_task = task
                result = {
                    "row_idx": int(row_idx),
                    "task": task,
                    "variation": row.get("variation"),
                    "episode": row.get("episode"),
                    "source_bundle": row.get("source_bundle"),
                    "split": row.get("split"),
                    "task_instruction": row.get("task_instruction"),
                    "wan_backend": args.wan_backend,
                    "wan_num_inference_steps": int(args.wan_num_inference_steps),
                    "policy_checkpoint": policy.checkpoint_dir.as_posix(),
                }
                try:
                    assert task_env is not None
                    result.update(run_episode(row_idx=row_idx, row=row, task_env=task_env, policy=policy, wan_provider=wan_provider, args=args))
                except Exception as exc:
                    result.update(
                        {
                            "success": False,
                            "terminal": False,
                            "invalid_actions": 0,
                            "executed_steps": 0,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(limit=10),
                        }
                    )
                    counts[task]["errors"] += 1

                counts[task]["n"] += 1
                counts[task]["success"] += int(result["success"])
                counts[task]["fail"] += int(not result["success"])
                counts[task]["invalid_actions"] += int(result.get("invalid_actions", 0))
                f.write(json.dumps(result, ensure_ascii=True) + "\n")
                f.flush()
                print(json.dumps({k: result[k] for k in ("row_idx", "task", "variation", "episode", "success", "executed_steps")}, sort_keys=True))
    finally:
        env.shutdown()

    summary = {
        "out": args.out.as_posix(),
        "num_episodes": len(rows),
        "success": int(sum(v["success"] for v in counts.values())),
        "counts": dict(counts),
    }
    summary["success_rate"] = summary["success"] / max(1, summary["num_episodes"])
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
