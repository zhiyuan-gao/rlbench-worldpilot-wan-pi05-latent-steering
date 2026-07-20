from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import pickle
import random
import traceback
from typing import Any
from urllib.error import HTTPError
from urllib import request as urlrequest

import numpy as np

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
from .rpc_numpy import decode_arrays, encode_arrays
from .sample_index import event_boundaries, read_jsonl, row_key


def parse_tasks(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    tasks: set[str] = set()
    for value in values:
        tasks.update(item for item in str(value).replace(",", " ").split() if item)
    return tasks or None


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


def _to_uint8_rgb(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr


def hstack_views(images: dict[str, Any], *, height: int = 256, view_width: int = 256) -> np.ndarray:
    from PIL import Image

    view_order = ("front", "left_shoulder", "right_shoulder")
    panels = []
    for name in view_order:
        if name not in images:
            continue
        im = Image.fromarray(_to_uint8_rgb(images[name]))
        im = im.resize((view_width, height), Image.BICUBIC)
        panels.append(np.asarray(im))
    if not panels:
        raise ValueError("No RGB views are available for hstack video recording.")
    return np.concatenate(panels, axis=1)


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> str | None:
    if not frames:
        return None
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path.as_posix(), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLBench rollout client for a separate WorldPilot policy RPC server.")
    parser.add_argument("--policy-url", default=os.environ.get("POLICY_RPC_URL", "http://127.0.0.1:8765/infer"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=os.environ.get("MANIFEST_PATH"), required=os.environ.get("MANIFEST_PATH") is None)
    parser.add_argument("--event-manifest-path", type=Path, default=os.environ.get("EVENT_MANIFEST_PATH"), required=os.environ.get("EVENT_MANIFEST_PATH") is None)
    parser.add_argument("--split", default=os.environ.get("SPLIT", "val"), choices=("train", "val", "test", "all"))
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=25)
    parser.add_argument("--selection", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wan-mode", choices=("matched", "off"), default="matched")
    parser.add_argument(
        "--wan-seed-mode",
        choices=("per_step", "per_event", "per_episode"),
        default="per_step",
    )

    parser.add_argument(
        "--rlbench-root",
        default=os.environ.get("RLBENCH_ROOT"),
        required=os.environ.get("RLBENCH_ROOT") is None,
    )
    parser.add_argument("--lowdim-root-200", type=Path, default=os.environ.get("LOWDIM_ROOT_200"))
    parser.add_argument("--lowdim-root-400", type=Path, default=os.environ.get("LOWDIM_ROOT_400"))
    parser.add_argument("--rgb-root-200", type=Path, default=os.environ.get("RGB_ROOT_200"))
    parser.add_argument("--rgb-root-400", type=Path, default=os.environ.get("RGB_ROOT_400"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)

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
    parser.add_argument("--rpc-timeout-sec", type=float, default=600.0)
    return parser.parse_args()


def root_for_source(row: dict[str, Any], root_200: Path, root_400: Path) -> Path:
    source = str(row.get("source_bundle", "all200"))
    if source in {"all200", "local200"}:
        return Path(root_200)
    if source in {"all400", "remote400"}:
        return Path(root_400)
    raise KeyError(f"Unsupported source_bundle={source!r}")


def infer_action7(args: argparse.Namespace, payload: dict[str, Any]) -> np.ndarray:
    body = pickle.dumps(encode_arrays(payload), protocol=pickle.HIGHEST_PROTOCOL)
    req = urlrequest.Request(args.policy_url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-python-pickle")
    try:
        with urlrequest.urlopen(req, timeout=float(args.rpc_timeout_sec)) as response:
            result = decode_arrays(pickle.loads(response.read()))
    except HTTPError as exc:
        try:
            result = decode_arrays(pickle.loads(exc.read()))
        except Exception as read_exc:
            raise RuntimeError(f"Policy RPC HTTP {exc.code}; could not decode error body: {read_exc!r}") from exc
    if not result.get("ok"):
        raise RuntimeError(f"Policy RPC failed: {result.get('error')}\n{result.get('traceback', '')}")
    return np.asarray(result["action7"], dtype=np.float32)


def run_episode_rpc(
    *,
    row_idx: int,
    row: dict[str, Any],
    task_env,
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
        episode_seed = int(args.seed) + row_idx * 1000
        action_seed = episode_seed + step_idx
        if args.wan_seed_mode == "per_step":
            wan_seed = episode_seed + step_idx
        elif args.wan_seed_mode == "per_event":
            wan_seed = episode_seed + event_idx
        else:
            wan_seed = episode_seed
        payload = {
            "current_images": current_images,
            "goal_images": goal_images,
            "policy_obs": policy_obs_from_rlbench(obs, prompt=prompt, gripper_threshold=args.state_gripper_threshold),
            "prompt": prompt,
            "wan_prompt": wan_prompt,
            "wan_mode": args.wan_mode,
            "wan_seed": wan_seed,
            "action_seed": action_seed,
        }
        action7 = infer_action7(args, payload)
        action9 = action7_to_rlbench_action9(action7, ignore_collision=args.ignore_collision)
        result = {
            "step_idx": int(step_idx),
            "event_idx": int(event_idx),
            "event_step_idx": int(event_step_idx),
            "event_end_frame": int(event_end),
            "wan_prompt": wan_prompt,
            "wan_mode": args.wan_mode,
            "wan_seed_mode": args.wan_seed_mode,
            "wan_seed": int(wan_seed),
            "action_seed": int(action_seed),
            "action7": np.asarray(action7, dtype=float).tolist(),
            "action9": action9.tolist(),
            "invalid_action": False,
        }
        try:
            obs, reward, terminal = task_env.step(action9)
        except Exception as exc:
            invalid_actions += 1
            result.update({"invalid_action": True, "error": repr(exc), "traceback": traceback.format_exc(limit=6)})
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
        elif args.event_switch_mode == "fixed_steps" and event_step_idx >= fixed_budget:
            switch_reason = "fixed_steps"
        elif args.event_switch_mode == "steps_only" and event_step_idx >= args.max_steps_per_event:
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
        "wan_mode": args.wan_mode,
        "wan_seed_mode": args.wan_seed_mode,
        "event_boundaries": [int(x) for x in boundaries],
        "video_path": video_path,
        "step_results": step_results,
    }


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    install_rlbench_root(args.rlbench_root)
    from rlbench.environment import Environment

    rows = merged_eval_rows(args)
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
                    "policy_url": args.policy_url,
                    "wan_mode": args.wan_mode,
                    "wan_seed_mode": args.wan_seed_mode,
                }
                try:
                    assert task_env is not None
                    result.update(run_episode_rpc(row_idx=row_idx, row=row, task_env=task_env, args=args))
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
                print(json.dumps({k: result[k] for k in ("row_idx", "task", "variation", "episode", "success", "executed_steps")}, sort_keys=True), flush=True)
    finally:
        env.shutdown()

    summary = {
        "out": args.out.as_posix(),
        "num_episodes": len(rows),
        "success": int(sum(v["success"] for v in counts.values())),
        "wan_mode": args.wan_mode,
        "wan_seed_mode": args.wan_seed_mode,
        "seed": args.seed,
        "counts": dict(counts),
    }
    summary["success_rate"] = summary["success"] / max(1, summary["num_episodes"])
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
