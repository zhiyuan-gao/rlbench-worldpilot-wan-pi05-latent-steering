from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import numpy as np
import safetensors.torch
import torch

from openpi import transforms as _transforms
from openpi.models import model as _model
import openpi.shared.normalize as _normalize
import openpi.training.config as _config

from .modeling import PI0WanLatentSteeringPytorch


def build_online_config(
    config_name: str,
    *,
    exp_name: str,
    checkpoint_base_dir: str | Path,
    assets_base_dir: str | Path,
    precision: str | None = None,
):
    config = _config.get_config(config_name)
    replace_kwargs = {
        "exp_name": exp_name,
        "checkpoint_base_dir": str(checkpoint_base_dir),
        "assets_base_dir": str(assets_base_dir),
    }
    if precision is not None:
        replace_kwargs["pytorch_training_precision"] = precision
    config = dataclasses.replace(config, **replace_kwargs)
    object.__setattr__(config.model, "dtype", config.pytorch_training_precision)
    return config


def resolve_checkpoint_dir(config, eval_checkpoint: str | Path | None) -> Path:
    if eval_checkpoint is None:
        checkpoint_dir = Path(config.checkpoint_dir)
        steps = [int(path.name) for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit()]
        if not steps:
            raise FileNotFoundError(f"No step checkpoints found under {checkpoint_dir}")
        return checkpoint_dir / str(max(steps))

    path = Path(eval_checkpoint)
    if path.is_file():
        if path.name != "model.safetensors":
            raise ValueError(f"Expected a model.safetensors file or checkpoint dir, got {path}")
        return path.parent
    if (path / "model.safetensors").exists():
        return path
    steps = [int(child.name) for child in path.iterdir() if child.is_dir() and child.name.isdigit()]
    if steps:
        return path / str(max(steps))
    raise FileNotFoundError(f"Could not resolve checkpoint directory from {path}")


def load_data_config_with_checkpoint_assets(config, checkpoint_dir: Path):
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.asset_id is None:
        return data_config
    ckpt_assets_dir = checkpoint_dir / "assets" / data_config.asset_id
    if (ckpt_assets_dir / "norm_stats.json").exists():
        norm_stats = _normalize.load(ckpt_assets_dir)
        return dataclasses.replace(data_config, norm_stats=norm_stats)
    return data_config


class WanPi05OnlinePolicy:
    def __init__(
        self,
        *,
        config_name: str,
        exp_name: str,
        checkpoint_base_dir: str | Path,
        assets_base_dir: str | Path,
        eval_checkpoint: str | Path | None,
        wan_latent_shape: tuple[int, int, int, int, int],
        device: str = "cuda",
        precision: str | None = None,
        wan_time_mode: str = "all",
        wan_num_heads: int = 8,
        wan_dropout: float = 0.0,
        action_num_steps: int = 10,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.action_num_steps = int(action_num_steps)

        self.config = build_online_config(
            config_name,
            exp_name=exp_name,
            checkpoint_base_dir=checkpoint_base_dir,
            assets_base_dir=assets_base_dir,
            precision=precision,
        )
        self.checkpoint_dir = resolve_checkpoint_dir(self.config, eval_checkpoint)
        self.data_config = load_data_config_with_checkpoint_assets(self.config, self.checkpoint_dir)
        if self.data_config.norm_stats is None:
            raise ValueError(
                "No norm stats found for online policy. Make sure the training checkpoint contains "
                "assets/<asset_id>/norm_stats.json or set ASSETS_BASE_DIR to a directory with stats."
            )

        self.model = PI0WanLatentSteeringPytorch(
            self.config.model,
            wan_time_mode=wan_time_mode,
            wan_num_heads=wan_num_heads,
            wan_dropout=wan_dropout,
        ).to(self.device)
        init_dtype = torch.bfloat16 if self.config.pytorch_training_precision == "bfloat16" else torch.float32
        dummy_latents = torch.zeros((1, *wan_latent_shape), dtype=init_dtype, device=self.device)
        self.model.initialize_wan_fuser(dummy_latents, device=self.device, dtype=init_dtype)
        safetensors.torch.load_model(
            self.model,
            self.checkpoint_dir / "model.safetensors",
            strict=True,
            device=str(self.device),
        )
        self.model.eval()

        self.input_transform = _transforms.compose(
            [
                *self.data_config.repack_transforms.inputs,
                *self.data_config.data_transforms.inputs,
                _transforms.Normalize(self.data_config.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *self.data_config.model_transforms.inputs,
            ]
        )
        self.output_transform = _transforms.compose(
            [
                *self.data_config.model_transforms.outputs,
                _transforms.Unnormalize(self.data_config.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *self.data_config.data_transforms.outputs,
                *self.data_config.repack_transforms.outputs,
            ]
        )

    def _to_batched_tensor(self, value: Any) -> torch.Tensor:
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
            raise TypeError(f"Online policy transform left a non-numeric value: {type(value)!r}")
        return torch.from_numpy(array).to(self.device)[None, ...]

    @torch.no_grad()
    def infer(self, obs: dict[str, Any], wan_latents: torch.Tensor | np.ndarray) -> dict[str, Any]:
        inputs = self.input_transform(dict(obs))
        inputs = jax.tree.map(self._to_batched_tensor, inputs)
        observation = _model.Observation.from_dict(inputs)
        wan_latents = torch.as_tensor(wan_latents, device=self.device)
        if wan_latents.ndim == 5:
            wan_latents = wan_latents[None]
        actions = self.model.sample_actions(
            self.device,
            observation,
            wan_latents=wan_latents,
            num_steps=self.action_num_steps,
        )
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        return self.output_transform(outputs)

    def infer_action7(self, obs: dict[str, Any], wan_latents: torch.Tensor | np.ndarray) -> np.ndarray:
        result = self.infer(obs, wan_latents)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            return actions[:7]
        return actions[0, :7]
