from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .latent_cache import load_latents
from .sample_index import VIEW_NAMES, build_sample_index, bundle_name, read_jsonl, write_jsonl


IGNORE_INDEX = -100
DATASET_STATS_KEY = "rlbench_selected10"


class DummyRlbenchObject:
    """Fallback class for reading RLBench/PyRep pickles without importing RLBench."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class LooseUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("rlbench") or module.startswith("pyrep"):
            return DummyRlbenchObject
        return super().find_class(module, name)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    xyz = q[..., :3]
    w = np.clip(q[..., 3], -1.0, 1.0)
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    scale = np.full_like(angle, 2.0)
    np.divide(angle, sin_half, out=scale, where=sin_half > 1e-8)
    return xyz * scale[..., None]


def gripper_open_value(obs: Any, threshold: float = 0.95) -> float:
    joint_positions = getattr(obs, "gripper_joint_positions", None)
    if joint_positions is None:
        return float(getattr(obs, "gripper_open", 0.0))
    joint_positions = np.asarray(joint_positions, dtype=np.float32)
    if joint_positions.size == 0:
        return float(getattr(obs, "gripper_open", 0.0))
    return 1.0 if float(joint_positions[0]) / 0.04 > float(threshold) else 0.0


def absolute_rotvec7_from_obs(obs: Any) -> np.ndarray:
    ee_pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    if ee_pose.shape != (7,):
        raise ValueError(f"Expected obs.gripper_pose shape (7,), got {ee_pose.shape}")
    rotvec = quat_to_rotvec(ee_pose[3:7]).astype(np.float32)
    return np.concatenate([ee_pose[:3], rotvec, [gripper_open_value(obs)]]).astype(np.float32)


def lowdim_roots(root_200: str | Path, root_400: str | Path) -> dict[str, Path]:
    return {"all200": Path(root_200).resolve(), "all400": Path(root_400).resolve()}


def lowdim_episode_dir(record: dict[str, Any], roots: dict[str, Path]) -> Path:
    bundle = bundle_name(str(record.get("source_bundle", "all200")))
    if bundle not in roots:
        raise KeyError(f"source_bundle={bundle!r} has no lowdim root. Known roots: {sorted(roots)}")
    rel = str(record["rgb_episode_relpath"])
    return roots[bundle] / rel


class ObservationCache:
    def __init__(self, roots: dict[str, Path]) -> None:
        self.roots = roots
        self._cache: dict[Path, Sequence[Any]] = {}

    def observations_for(self, record: dict[str, Any]) -> Sequence[Any]:
        episode_dir = lowdim_episode_dir(record, self.roots)
        if episode_dir not in self._cache:
            path = episode_dir / "low_dim_obs.pkl"
            with path.open("rb") as f:
                demo = LooseUnpickler(f).load()
            observations = getattr(demo, "_observations", demo)
            if not isinstance(observations, (list, tuple)):
                raise ValueError(f"{path} did not contain a sequence of observations")
            self._cache[episode_dir] = observations
        return self._cache[episode_dir]


@dataclass(frozen=True)
class BoundsStats:
    mean: list[float]
    std: list[float]
    min: list[float]
    max: list[float]
    q01: list[float]
    q99: list[float]
    mask: list[bool]

    @classmethod
    def from_array(cls, values: np.ndarray) -> "BoundsStats":
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Expected values with shape [N,D], got {values.shape}")
        return cls(
            mean=np.mean(values, axis=0).astype(float).tolist(),
            std=np.std(values, axis=0).astype(float).tolist(),
            min=np.min(values, axis=0).astype(float).tolist(),
            max=np.max(values, axis=0).astype(float).tolist(),
            q01=np.quantile(values, 0.01, axis=0).astype(float).tolist(),
            q99=np.quantile(values, 0.99, axis=0).astype(float).tolist(),
            mask=[True] * values.shape[1],
        )

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        q01 = np.asarray(self.q01, dtype=np.float32)
        q99 = np.asarray(self.q99, dtype=np.float32)
        mask = np.asarray(self.mask, dtype=bool)
        denom = np.maximum(q99 - q01, 1e-6)
        normalized = 2.0 * (values - q01) / denom - 1.0
        normalized = np.where(mask, normalized, values)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class RLBenchOpenVLAStats:
    action: BoundsStats
    proprio: BoundsStats

    def to_json(self) -> dict[str, Any]:
        return {
            DATASET_STATS_KEY: {
                "action": self.action.__dict__,
                "proprio": self.proprio.__dict__,
            }
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "RLBenchOpenVLAStats":
        if DATASET_STATS_KEY in payload:
            payload = payload[DATASET_STATS_KEY]
        return cls(
            action=BoundsStats(**payload["action"]),
            proprio=BoundsStats(**payload["proprio"]),
        )


def build_or_load_sample_index(
    *,
    sample_index_path: str | Path,
    manifest_path: str | Path,
    event_manifest_path: str | Path | None,
    goal_mode: str,
    split: str,
    sample_every_n: int,
    rgb_root_200: str | Path,
    rgb_root_400: str | Path,
    rebuild: bool = False,
) -> list[dict[str, Any]]:
    sample_index_path = Path(sample_index_path)
    if sample_index_path.exists() and not rebuild:
        return read_jsonl(sample_index_path)
    rows = build_sample_index(
        manifest_path=Path(manifest_path),
        event_manifest_path=Path(event_manifest_path) if event_manifest_path else None,
        goal_mode=goal_mode,
        split=split,
        sample_every_n=sample_every_n,
        rgb_root_200=rgb_root_200,
        rgb_root_400=rgb_root_400,
    )
    write_jsonl(sample_index_path, rows)
    return rows


def fit_or_load_stats(
    *,
    stats_path: str | Path,
    sample_index: Sequence[dict[str, Any]],
    lowdim_root_200: str | Path,
    lowdim_root_400: str | Path,
    force_refit: bool = False,
) -> RLBenchOpenVLAStats:
    stats_path = Path(stats_path)
    if stats_path.exists() and not force_refit:
        return RLBenchOpenVLAStats.from_json(json.loads(stats_path.read_text(encoding="utf-8")))

    obs_cache = ObservationCache(lowdim_roots(lowdim_root_200, lowdim_root_400))
    actions = []
    proprios = []
    for record in sample_index:
        observations = obs_cache.observations_for(record)
        current = int(record["frame_index"])
        target = int(record["target_waypoint_frame"])
        proprios.append(absolute_rotvec7_from_obs(observations[current]))
        actions.append(absolute_rotvec7_from_obs(observations[target]))

    stats = RLBenchOpenVLAStats(
        action=BoundsStats.from_array(np.asarray(actions, dtype=np.float32)),
        proprio=BoundsStats.from_array(np.asarray(proprios, dtype=np.float32)),
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    return stats


class OpenVLAOFTRLBenchDataset(Dataset):
    """RLBench selected10 waypoint samples formatted for OpenVLA-OFT L1 training."""

    def __init__(
        self,
        sample_index: Sequence[dict[str, Any]],
        *,
        lowdim_root_200: str | Path,
        lowdim_root_400: str | Path,
        stats: RLBenchOpenVLAStats,
        action_tokenizer: Callable[[np.ndarray], str | list[str]] | None = None,
        base_tokenizer: Any | None = None,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
        prompt_builder_cls: type | None = None,
        num_images_in_input: int = 3,
        num_actions_chunk: int = 8,
        action_dim: int = 7,
        use_proprio: bool = True,
        predict_stop_token: bool = True,
        use_wan_latents: bool = False,
        wan_latent_cache_root: str | Path | None = None,
        expected_wan_num_inference_steps: int | None = None,
        expected_wan_backend: str | None = None,
        expected_wan_shape: tuple[int, int, int, int, int] | None = None,
        allow_missing_wan_latents: bool = False,
    ) -> None:
        if num_images_in_input < 1 or num_images_in_input > len(VIEW_NAMES):
            raise ValueError(f"num_images_in_input must be 1..{len(VIEW_NAMES)}, got {num_images_in_input}")
        if action_dim != 7:
            raise ValueError(f"RLBench absolute rotvec action_dim must be 7, got {action_dim}")
        self.sample_index = list(sample_index)
        self.obs_cache = ObservationCache(lowdim_roots(lowdim_root_200, lowdim_root_400))
        self.stats = stats
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_cls = prompt_builder_cls
        self.num_images_in_input = int(num_images_in_input)
        self.num_actions_chunk = int(num_actions_chunk)
        self.action_dim = int(action_dim)
        self.use_proprio = bool(use_proprio)
        self.predict_stop_token = bool(predict_stop_token)
        self.use_wan_latents = bool(use_wan_latents)
        self.wan_latent_cache_root = Path(wan_latent_cache_root) if wan_latent_cache_root else None
        self.expected_wan_num_inference_steps = expected_wan_num_inference_steps
        self.expected_wan_backend = expected_wan_backend
        self.expected_wan_shape = expected_wan_shape
        self.allow_missing_wan_latents = bool(allow_missing_wan_latents)

        if self.use_wan_latents and self.wan_latent_cache_root is None:
            raise ValueError("wan_latent_cache_root is required when use_wan_latents=True")

    def __len__(self) -> int:
        return len(self.sample_index)

    def raw_example(self, index: int) -> dict[str, Any]:
        record = self.sample_index[index]
        observations = self.obs_cache.observations_for(record)
        current = int(record["frame_index"])
        target = int(record["target_waypoint_frame"])
        action = absolute_rotvec7_from_obs(observations[target])
        proprio = absolute_rotvec7_from_obs(observations[current])
        action_norm = self.stats.action.normalize(action)
        proprio_norm = self.stats.proprio.normalize(proprio)
        action_chunk = np.repeat(action_norm[None], self.num_actions_chunk, axis=0).astype(np.float32)
        image_paths = record.get("current_image_paths")
        if not isinstance(image_paths, dict):
            raise KeyError("sample_index row does not contain current_image_paths; rebuild it with RGB roots")
        return {
            "record": record,
            "action": action.astype(np.float32),
            "proprio": proprio.astype(np.float32),
            "actions": action_chunk,
            "proprio_normalized": proprio_norm.astype(np.float32),
            "image_paths": {view: str(image_paths[view]) for view in VIEW_NAMES[: self.num_images_in_input]},
            "language": str(record.get("task_instruction") or record.get("task") or "").strip().lower(),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.raw_example(index)
        if self.action_tokenizer is None or self.base_tokenizer is None or self.image_transform is None:
            return raw
        if self.prompt_builder_cls is None:
            raise ValueError("prompt_builder_cls is required for tokenized OpenVLA-OFT examples")

        views = VIEW_NAMES[: self.num_images_in_input]
        pixels = []
        for view in views:
            img = Image.open(raw["image_paths"][view]).convert("RGB")
            pixels.append(self.image_transform(img))

        current_action_string = self.action_tokenizer(raw["actions"][0])
        future_action_string = "".join(self.action_tokenizer(raw["actions"][1:]))
        action_chunk_string = current_action_string + future_action_string

        prompt_builder = self.prompt_builder_cls("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {raw['language']}?")
        prompt_builder.add_turn("gpt", action_chunk_string)

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        action_chunk_len = len(action_chunk_string)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        labels_t[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels_t[-1] = IGNORE_INDEX

        output = {
            "pixel_values": pixels[0],
            "input_ids": input_ids_t,
            "labels": labels_t,
            "dataset_name": DATASET_STATS_KEY,
            "actions": raw["actions"],
        }
        if len(pixels) > 1:
            output["pixel_values_wrist"] = torch.cat(pixels[1:], dim=0)
        if self.use_proprio:
            output["proprio"] = raw["proprio_normalized"]
        if self.use_wan_latents:
            assert self.wan_latent_cache_root is not None
            output["wan_latents"] = load_latents(
                self.wan_latent_cache_root,
                raw["record"],
                allow_missing=self.allow_missing_wan_latents,
                expected_num_inference_steps=self.expected_wan_num_inference_steps,
                expected_backend=self.expected_wan_backend,
                expected_shape=self.expected_wan_shape,
            )
        return output


class WanLatentCollator:
    def __init__(self, base_collator: Callable[[Sequence[dict[str, Any]]], dict[str, Any]]) -> None:
        self.base_collator = base_collator

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
        output = self.base_collator(instances)
        if "wan_latents" in instances[0]:
            output["wan_latents"] = torch.stack([instance["wan_latents"] for instance in instances])
        return output
