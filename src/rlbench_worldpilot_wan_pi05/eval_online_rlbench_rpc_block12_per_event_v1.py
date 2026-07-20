from __future__ import annotations

import argparse
import hashlib
import traceback
from typing import Any

import numpy as np

from . import eval_online_rlbench_rpc as _legacy
from .rlbench_online_utils import (
    action7_to_rlbench_action9,
    load_demo,
    obs_rgb_images,
    policy_obs_from_rlbench,
    pose_event_reached,
    read_goal_images_from_episode,
    resolve_episode_dir,
)
from .sample_index import event_boundaries


PROTOCOL_VERSION = "block12_per_event_v1"
_SEED_MODULUS = 2**63 - 1


def episode_uid(row: dict[str, Any]) -> str:
    return "/".join(str(row[key]) for key in ("task", "variation", "episode"))


def stable_seed(*, eval_seed: int, uid: str, stream: str, index: int) -> int:
    payload = f"{PROTOCOL_VERSION}\0{int(eval_seed)}\0{uid}\0{stream}\0{int(index)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % _SEED_MODULUS


def advance_event(
    *,
    event_idx: int,
    num_events: int,
    event_step_idx: int,
    switch_reason: str | None,
    final_event_locked: bool,
) -> tuple[int, int, bool]:
    if switch_reason is None or final_event_locked:
        return event_idx, event_step_idx, final_event_locked
    if event_idx < num_events - 1:
        return event_idx + 1, 0, False
    return event_idx, 0, True


def validate_frozen_protocol(args: argparse.Namespace) -> None:
    expected = {
        "split": "val",
        "max_episodes_per_task": 25,
        "selection": "first",
        "wan_mode": "matched",
        "wan_seed_mode": "per_event",
        "wan_text_source": "task",
        "event_switch_mode": "pose_or_steps",
        "max_steps": 30,
        "max_steps_per_event": 8,
        "fixed_steps_per_event": None,
        "event_goal_pos_threshold": 0.04,
        "event_goal_rot_threshold": 0.5,
        "event_goal_gripper_threshold": 0.5,
        "continue_on_invalid": False,
    }
    mismatches = {
        name: {"expected": expected_value, "actual": getattr(args, name)}
        for name, expected_value in expected.items()
        if getattr(args, name) != expected_value
    }
    if args.seed not in (0, 1, 2):
        mismatches["seed"] = {"expected": [0, 1, 2], "actual": args.seed}
    if mismatches:
        raise ValueError(f"Arguments do not match {PROTOCOL_VERSION}: {mismatches}")


def run_episode_rpc(
    *,
    row_idx: int,
    row: dict[str, Any],
    task_env,
    args: argparse.Namespace,
) -> dict[str, Any]:
    del row_idx  # Seeds are intentionally independent of task selection and row ordering.
    lowdim_root = _legacy.root_for_source(row, args.lowdim_root_200, args.lowdim_root_400)
    rgb_root = _legacy.root_for_source(row, args.rgb_root_200, args.rgb_root_400)
    lowdim_episode_dir = resolve_episode_dir(row, lowdim_root)
    rgb_episode_dir = resolve_episode_dir(row, rgb_root)

    demo = load_demo(lowdim_episode_dir)
    observations = getattr(demo, "_observations", demo)
    boundaries = event_boundaries(row, int(row.get("num_frames", len(observations))))
    num_events = len(boundaries) - 1
    if num_events < 1:
        raise ValueError(f"Episode has no events: {episode_uid(row)}")
    _, obs = task_env.reset_to_demo(demo)

    frames: list[np.ndarray] = []
    if args.record_video:
        frames.append(
            np.asarray(_legacy.hstack_views(obs_rgb_images(obs), height=args.image_size, view_width=args.image_size))
        )

    uid = episode_uid(row)
    event_idx = 0
    event_step_idx = 0
    final_event_locked = False
    success = False
    terminal = False
    invalid_actions = 0
    step_results = []
    goal_images = None
    goal_frame = None
    goal_obs = None

    for step_idx in range(int(args.max_steps)):
        event_end = int(boundaries[event_idx + 1])
        if goal_frame != event_end:
            goal_frame = event_end
            goal_images = read_goal_images_from_episode(rgb_episode_dir, goal_frame)
            goal_obs = observations[goal_frame]
        assert goal_images is not None and goal_obs is not None

        current_images = obs_rgb_images(obs)
        prompt = str(row.get("task_instruction") or row.get("task") or "")
        wan_prompt = _legacy.wan_prompt_for_event(row, event_idx, args.wan_text_source)
        action_seed = stable_seed(eval_seed=args.seed, uid=uid, stream="action", index=step_idx)
        wan_seed = stable_seed(eval_seed=args.seed, uid=uid, stream="wan", index=event_idx)
        payload = {
            "current_images": current_images,
            "goal_images": goal_images,
            "policy_obs": policy_obs_from_rlbench(
                obs,
                prompt=prompt,
                gripper_threshold=args.state_gripper_threshold,
            ),
            "prompt": prompt,
            "wan_prompt": wan_prompt,
            "wan_mode": args.wan_mode,
            "wan_seed": wan_seed,
            "action_seed": action_seed,
        }
        action7 = _legacy.infer_action7(args, payload)
        action9 = action7_to_rlbench_action9(action7, ignore_collision=args.ignore_collision)
        result = {
            "protocol_version": PROTOCOL_VERSION,
            "episode_uid": uid,
            "step_idx": int(step_idx),
            "event_idx": int(event_idx),
            "event_step_idx": int(event_step_idx),
            "event_end_frame": int(event_end),
            "final_event_locked_before_step": bool(final_event_locked),
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
        if not final_event_locked:
            if reached:
                switch_reason = "pose_reached"
            elif event_step_idx >= args.max_steps_per_event:
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
            frames.append(
                np.asarray(_legacy.hstack_views(obs_rgb_images(obs), height=args.image_size, view_width=args.image_size))
            )
        if success:
            break
        event_idx, event_step_idx, final_event_locked = advance_event(
            event_idx=event_idx,
            num_events=num_events,
            event_step_idx=event_step_idx,
            switch_reason=switch_reason,
            final_event_locked=final_event_locked,
        )

    video_path = None
    if args.record_video and frames and (not args.record_failures_only or not success):
        video_path = _legacy.write_video(
            args.out.parent / "videos" / f"{uid.replace('/', '__')}.mp4",
            frames,
            fps=args.video_fps,
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "episode_uid": uid,
        "success": bool(success),
        "terminal": bool(terminal),
        "invalid_actions": int(invalid_actions),
        "executed_steps": len(step_results),
        "event_switch_mode": args.event_switch_mode,
        "wan_mode": args.wan_mode,
        "wan_seed_mode": args.wan_seed_mode,
        "event_boundaries": [int(x) for x in boundaries],
        "final_event_locked": bool(final_event_locked),
        "video_path": video_path,
        "step_results": step_results,
    }


_LEGACY_PARSE_ARGS = _legacy.parse_args


def _parse_and_validate_args() -> argparse.Namespace:
    args = _LEGACY_PARSE_ARGS()
    validate_frozen_protocol(args)
    return args


def main() -> None:
    _legacy.parse_args = _parse_and_validate_args
    _legacy.run_episode_rpc = run_episode_rpc
    _legacy.main()


if __name__ == "__main__":
    main()
