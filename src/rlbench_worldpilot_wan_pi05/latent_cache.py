from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .sample_index import cache_relpath


def latent_path_for_record(cache_root: str | Path, record: dict[str, Any]) -> Path:
    relpath = str(record.get("latent_relpath") or cache_relpath(record))
    return Path(cache_root) / relpath


def save_latent_record(
    path: str | Path,
    latents: torch.Tensor,
    *,
    record: dict[str, Any],
    latent_layout: str = "vcthw",
    backend: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "future_video_latents": latents.detach().cpu(),
        "latent_layout": latent_layout,
        "view_names": ["front", "left_shoulder", "right_shoulder"],
        "record": record,
        "backend": backend,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_latents(
    cache_root: str | Path,
    record: dict[str, Any],
    *,
    allow_missing: bool = False,
    dummy_shape: tuple[int, int, int, int, int] = (3, 16, 6, 32, 32),
    dummy_dtype: torch.dtype = torch.float16,
    expected_num_inference_steps: int | None = None,
    expected_backend: str | None = None,
) -> torch.Tensor:
    path = latent_path_for_record(cache_root, record)
    if not path.exists():
        if allow_missing:
            return torch.zeros(dummy_shape, dtype=dummy_dtype)
        raise FileNotFoundError(f"Missing WAN latent cache for lerobot_index={record.get('lerobot_index')}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        if expected_num_inference_steps is not None or expected_backend is not None:
            raise ValueError(f"{path} is a raw tensor cache without metadata; cannot verify WAN cache version")
        latents = payload
    elif isinstance(payload, dict):
        metadata = payload.get("metadata") or {}
        if expected_num_inference_steps is not None:
            actual_steps = metadata.get("num_inference_steps")
            if actual_steps is None or int(actual_steps) != int(expected_num_inference_steps):
                raise ValueError(
                    f"{path} has num_inference_steps={actual_steps}, "
                    f"expected {expected_num_inference_steps}"
                )
        if expected_backend is not None:
            actual_backend = payload.get("backend")
            if actual_backend != expected_backend:
                raise ValueError(f"{path} has backend={actual_backend!r}, expected {expected_backend!r}")
        latents = payload.get("future_video_latents")
        if latents is None:
            latents = payload.get("latents")
        if latents is None:
            raise KeyError(f"{path} does not contain future_video_latents")
    else:
        raise TypeError(f"Unsupported latent payload type in {path}: {type(payload)!r}")
    if latents.ndim != 5:
        raise ValueError(f"Expected per-sample latents [V,C,T,H,W], got {tuple(latents.shape)} from {path}")
    return latents
