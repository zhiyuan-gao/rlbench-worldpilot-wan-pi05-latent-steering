from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch

from openpi.models import model as _model
from openpi.training import data_loader as _openpi_data

from .latent_cache import load_latents
from .sample_index import build_sample_index, read_jsonl


class WanLatentAlignedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        sample_index: list[dict[str, Any]],
        *,
        latent_cache_root: str | Path,
        allow_missing_latents: bool = False,
        dummy_latent_shape: tuple[int, int, int, int, int] = (3, 16, 6, 32, 32),
        expected_num_inference_steps: int | None = None,
        expected_backend: str | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.sample_index = sample_index
        self.latent_cache_root = Path(latent_cache_root)
        self.allow_missing_latents = allow_missing_latents
        self.dummy_latent_shape = dummy_latent_shape
        self.expected_num_inference_steps = expected_num_inference_steps
        self.expected_backend = expected_backend
        if len(self.base_dataset) != len(self.sample_index):
            raise ValueError(
                f"LeRobot dataset length ({len(self.base_dataset)}) does not match sample index "
                f"length ({len(self.sample_index)}). Check split/sample_every_n."
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base_dataset[index])
        record = self.sample_index[index]
        sample["wan_latents"] = load_latents(
            self.latent_cache_root,
            record,
            allow_missing=self.allow_missing_latents,
            dummy_shape=self.dummy_latent_shape,
            expected_num_inference_steps=self.expected_num_inference_steps,
            expected_backend=self.expected_backend,
        ).numpy()
        sample["lerobot_index"] = np.asarray(int(record["lerobot_index"]), dtype=np.int64)
        return sample


def stack_tree(items):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def collate_wan_latent_batch(items):
    batch = stack_tree(items)
    wan_latents = torch.as_tensor(batch.pop("wan_latents"))
    lerobot_index = torch.as_tensor(batch.pop("lerobot_index"))
    batch = jax.tree.map(torch.as_tensor, batch)
    observation = _model.Observation.from_dict(batch)
    actions = torch.as_tensor(batch["actions"])
    return observation, actions, wan_latents, lerobot_index


def build_or_load_sample_index(
    *,
    sample_index_path: str | Path,
    manifest_path: str | Path,
    event_manifest_path: str | Path | None,
    goal_mode: str,
    split: str,
    sample_every_n: int,
    rgb_root_200: str | Path | None = None,
    rgb_root_400: str | Path | None = None,
    rebuild: bool = False,
) -> list[dict[str, Any]]:
    sample_index_path = Path(sample_index_path)
    if sample_index_path.exists() and not rebuild:
        return read_jsonl(sample_index_path)
    rows = build_sample_index(
        Path(manifest_path),
        event_manifest_path=Path(event_manifest_path) if event_manifest_path is not None else None,
        goal_mode=goal_mode,
        split=split,
        sample_every_n=sample_every_n,
        rgb_root_200=rgb_root_200,
        rgb_root_400=rgb_root_400,
    )
    from .sample_index import write_jsonl

    write_jsonl(sample_index_path, rows)
    return rows


def create_openpi_transformed_dataset(config, *, skip_norm_stats: bool = False):
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _openpi_data.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _openpi_data.transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    return dataset, data_config


def create_wan_latent_loader(
    config,
    *,
    manifest_path: str | Path,
    event_manifest_path: str | Path | None,
    goal_mode: str,
    sample_index_path: str | Path,
    latent_cache_root: str | Path,
    split: str,
    sample_every_n: int,
    rgb_root_200: str | Path | None,
    rgb_root_400: str | Path | None,
    local_batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    allow_missing_latents: bool = False,
    expected_num_inference_steps: int | None = None,
    expected_backend: str | None = None,
    rebuild_sample_index: bool = False,
    skip_norm_stats: bool = False,
):
    base_dataset, data_config = create_openpi_transformed_dataset(config, skip_norm_stats=skip_norm_stats)
    sample_index = build_or_load_sample_index(
        sample_index_path=sample_index_path,
        manifest_path=manifest_path,
        event_manifest_path=event_manifest_path,
        goal_mode=goal_mode,
        split=split,
        sample_every_n=sample_every_n,
        rgb_root_200=rgb_root_200,
        rgb_root_400=rgb_root_400,
        rebuild=rebuild_sample_index,
    )
    dataset = WanLatentAlignedDataset(
        base_dataset,
        sample_index,
        latent_cache_root=latent_cache_root,
        allow_missing_latents=allow_missing_latents,
        expected_num_inference_steps=expected_num_inference_steps,
        expected_backend=expected_backend,
    )
    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            drop_last=True,
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_wan_latent_batch,
        drop_last=True,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    logging.info("Created WAN latent loader with %d samples, local_batch_size=%d", len(dataset), local_batch_size)
    return loader, data_config
