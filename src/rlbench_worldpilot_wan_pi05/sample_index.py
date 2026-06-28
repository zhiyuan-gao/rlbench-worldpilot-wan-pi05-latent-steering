from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


VIEW_NAMES = ("front", "left_shoulder", "right_shoulder")
GoalMode = str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def clean_waypoints(row: dict[str, Any], num_frames: int) -> list[int]:
    raw = row.get("full_task_heuristic_waypoints")
    if raw is None:
        raw = row.get("clean_keypoints") or row.get("keypoints") or row.get("event_all_grouping_keypoints")
    if raw is None:
        return []
    points = []
    for point in raw:
        value = int(point)
        if 0 <= value < num_frames:
            points.append(value)
    return sorted(set(points))


def row_key(row: dict[str, Any]) -> str:
    if row.get("split_unique_id"):
        return str(row["split_unique_id"])
    return "__".join(
        str(row.get(name, ""))
        for name in ("source_bundle", "task", "variation", "episode")
    )


def build_event_lookup(event_manifest_path: Path | None) -> dict[str, dict[str, Any]]:
    if event_manifest_path is None:
        return {}
    return {row_key(row): row for row in read_jsonl(event_manifest_path)}


def event_boundaries(row: dict[str, Any], num_frames: int) -> list[int]:
    raw = row.get("event_keyframes_adjusted")
    if raw is None:
        grouped = row.get("event_heuristic_waypoints_by_event")
        if grouped is not None:
            raw = [0] + [group[-1] for group in grouped if group]
    if raw is None:
        raise KeyError(f"Event row {row_key(row)!r} has no event_keyframes_adjusted")

    points = []
    for point in raw:
        value = int(point)
        if 0 <= value < num_frames:
            points.append(value)
    if not points or points[0] != 0:
        points.insert(0, 0)
    if points[-1] != num_frames - 1:
        points.append(num_frames - 1)
    points = sorted(set(points))
    if len(points) < 2:
        raise ValueError(f"Event row {row_key(row)!r} has fewer than 2 event boundaries: {points}")
    return points


def event_goal_for_current(current: int, boundaries: list[int]) -> tuple[int, int, int]:
    """Return (event_idx, event_start, event_end) for a current frame.

    Event intervals are left-closed and right-open, so a frame exactly at an
    event boundary belongs to the next event. This keeps waypoint samples such
    as current=event_end aligned with the next event's goal.
    """
    current = int(current)
    for event_idx in range(len(boundaries) - 1):
        start = int(boundaries[event_idx])
        end = int(boundaries[event_idx + 1])
        if start <= current < end:
            return event_idx, start, end
    last_idx = len(boundaries) - 2
    return last_idx, int(boundaries[last_idx]), int(boundaries[-1])


def current_frames_for_segment(start: int, target: int, sample_every_n: int) -> list[int]:
    if sample_every_n <= 0:
        return [int(start)]
    frames = list(range(int(start), int(target), int(sample_every_n)))
    if int(start) not in frames:
        frames.insert(0, int(start))
    return sorted(set(f for f in frames if int(start) <= f < int(target)))


def bundle_name(source_bundle: str) -> str:
    if source_bundle in {"all200", "local200"}:
        return "all200"
    if source_bundle in {"all400", "remote400"}:
        return "all400"
    return str(source_bundle)


def image_path(root: str | Path, rgb_episode_relpath: str, view: str, frame: int) -> str:
    return (Path(root) / rgb_episode_relpath / f"{view}_rgb" / f"{int(frame)}.png").as_posix()


def roots_for_bundle(record: dict[str, Any], rgb_root_200: str | Path, rgb_root_400: str | Path) -> Path:
    bundle = bundle_name(str(record.get("source_bundle", "all200")))
    if bundle == "all200":
        return Path(rgb_root_200)
    if bundle == "all400":
        return Path(rgb_root_400)
    raise KeyError(f"Unsupported source_bundle={record.get('source_bundle')!r}")


def cache_relpath(record: dict[str, Any]) -> str:
    task = str(record["task"])
    source = str(record["source_bundle"])
    variation = str(record["variation"])
    episode = str(record["episode"])
    segment_idx = int(record["segment_idx"])
    frame = int(record["frame_index"])
    action_target = int(record["target_waypoint_frame"])
    latent_goal = int(record.get("latent_goal_frame", action_target))
    return (
        f"{record.get('split', 'train')}/{task}/"
        f"{source}__{variation}__{episode}__seg{segment_idx:03d}"
        f"__cur{frame:06d}__act{action_target:06d}__goal{latent_goal:06d}.pt"
    )


def build_sample_index(
    manifest_path: Path,
    *,
    event_manifest_path: Path | None = None,
    goal_mode: GoalMode = "event_end",
    split: str = "train",
    sample_every_n: int = 0,
    rgb_root_200: str | Path | None = None,
    rgb_root_400: str | Path | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    if goal_mode not in {"event_end", "next_waypoint"}:
        raise ValueError(f"Unsupported goal_mode={goal_mode!r}")
    event_lookup = build_event_lookup(event_manifest_path)
    if goal_mode == "event_end" and not event_lookup:
        raise ValueError("--event-manifest-path is required when --goal-mode=event_end")

    rows = []
    lerobot_index = 0
    for episode_row in read_jsonl(manifest_path):
        if split != "all" and str(episode_row.get("split")) != split:
            continue
        num_frames = int(episode_row.get("num_frames", 0))
        waypoints = clean_waypoints(episode_row, num_frames)
        points = [0] + [p for p in waypoints if 0 < int(p) < num_frames]
        points = sorted(set(int(p) for p in points))
        if len(points) < 2:
            continue

        boundaries = None
        event_row = None
        if goal_mode == "event_end":
            event_row = event_lookup.get(row_key(episode_row))
            if event_row is None:
                raise KeyError(f"No event manifest row found for {row_key(episode_row)!r}")
            boundaries = event_boundaries(event_row, num_frames)

        task_text = str(episode_row.get("task_instruction") or episode_row.get("task") or "").strip()
        rgb_episode_relpath = str(episode_row["rgb_episode_relpath"])
        for segment_idx in range(len(points) - 1):
            start = int(points[segment_idx])
            target = int(points[segment_idx + 1])
            if target <= start:
                continue
            for current in current_frames_for_segment(start, target, sample_every_n):
                if goal_mode == "event_end":
                    assert boundaries is not None
                    event_idx, event_start, latent_goal = event_goal_for_current(current, boundaries)
                    event_description = None
                    event_descriptions = event_row.get("event_descriptions") if event_row is not None else None
                    if isinstance(event_descriptions, list) and 0 <= event_idx < len(event_descriptions):
                        event_description = event_descriptions[event_idx]
                else:
                    event_idx = segment_idx
                    event_start = start
                    latent_goal = target
                    event_description = None
                record = {
                    "lerobot_index": int(lerobot_index),
                    "split": episode_row.get("split"),
                    "source_bundle": bundle_name(str(episode_row.get("source_bundle", "all200"))),
                    "source_dataset_id": episode_row.get("source_dataset_id"),
                    "rgb_episode_relpath": rgb_episode_relpath,
                    "task": episode_row.get("task"),
                    "variation": episode_row.get("variation"),
                    "variation_id": episode_row.get("variation_id"),
                    "episode": episode_row.get("episode"),
                    "episode_id": episode_row.get("episode_id"),
                    "segment_idx": int(segment_idx),
                    "frame_index": int(current),
                    "current_frame_idx": int(current),
                    "action_target_waypoint_frame": int(target),
                    "action_target_frame_idx": int(target),
                    "target_waypoint_frame": int(target),
                    "target_frame_idx": int(target),
                    "latent_goal_mode": str(goal_mode),
                    "latent_goal_frame": int(latent_goal),
                    "latent_goal_frame_idx": int(latent_goal),
                    "event_idx": int(event_idx),
                    "event_start_frame": int(event_start),
                    "event_end_frame": int(latent_goal),
                    "event_description": event_description,
                    "task_instruction": task_text,
                }
                if rgb_root_200 is not None and rgb_root_400 is not None:
                    root = roots_for_bundle(record, rgb_root_200, rgb_root_400)
                    record["current_image_paths"] = {
                        view: image_path(root, rgb_episode_relpath, view, current) for view in VIEW_NAMES
                    }
                    record["action_target_image_paths"] = {
                        view: image_path(root, rgb_episode_relpath, view, target) for view in VIEW_NAMES
                    }
                    record["latent_goal_image_paths"] = {
                        view: image_path(root, rgb_episode_relpath, view, latent_goal) for view in VIEW_NAMES
                    }
                    record["target_image_paths"] = record["latent_goal_image_paths"]
                record["latent_relpath"] = cache_relpath(record)
                rows.append(record)
                lerobot_index += 1
                if max_samples is not None and len(rows) >= max_samples:
                    return rows
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build row-aligned sample index for RLBench pi0.5/WAN latent cache.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--event-manifest-path", type=Path, default=None)
    parser.add_argument("--goal-mode", choices=("event_end", "next_waypoint"), default="event_end")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test", "all"))
    parser.add_argument("--sample-every-n", type=int, default=0)
    parser.add_argument("--rgb-root-200", type=Path, default=None)
    parser.add_argument("--rgb-root-400", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_sample_index(
        args.manifest_path,
        event_manifest_path=args.event_manifest_path,
        goal_mode=args.goal_mode,
        split=args.split,
        sample_every_n=args.sample_every_n,
        rgb_root_200=args.rgb_root_200,
        rgb_root_400=args.rgb_root_400,
        max_samples=args.max_samples,
    )
    write_jsonl(args.out, rows)
    print(json.dumps({"out": args.out.as_posix(), "num_samples": len(rows), "split": args.split}, sort_keys=True))


if __name__ == "__main__":
    main()
