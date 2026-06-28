from __future__ import annotations

import inspect
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image
import torch

from .export_wan_latent_cache import as_tensor_from_pipeline_output, normalize_hstack_latents, torch_dtype
from .sample_index import VIEW_NAMES


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image.shape}")
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if np.nanmax(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def hstack_views(
    images_by_view: Mapping[str, np.ndarray],
    *,
    height: int,
    view_width: int,
    num_views: int = 3,
) -> Image.Image:
    canvas = Image.new("RGB", (view_width * num_views, height))
    for view_idx, view in enumerate(VIEW_NAMES[:num_views]):
        image = Image.fromarray(to_uint8_rgb(images_by_view[view])).resize((view_width, height), Image.BICUBIC)
        canvas.paste(image, (view_idx * view_width, 0))
    return canvas


class DummyWanLatentProvider:
    def __init__(self, shape: tuple[int, int, int, int, int] = (3, 16, 6, 32, 32)) -> None:
        self.shape = tuple(int(x) for x in shape)

    def __call__(
        self,
        current_images: Mapping[str, np.ndarray],
        goal_images: Mapping[str, np.ndarray],
        *,
        prompt: str,
        seed: int = 0,
    ) -> torch.Tensor:
        del current_images, goal_images, prompt, seed
        return torch.zeros(self.shape, dtype=torch.float16)


class WanDiffusersOnlineProvider:
    """Online WAN FLF latent provider.

    This mirrors the cache exporter: current multi-view RGB and fixed event-goal
    multi-view RGB are horizontally stacked, sent through WAN FLF, and returned
    as per-view latents with shape ``[V,C,T,H,W]``.
    """

    def __init__(
        self,
        *,
        base_model: str | Path,
        lora_dir: str | Path | None = None,
        height: int = 256,
        view_width: int = 256,
        num_views: int = 3,
        num_frames: int = 21,
        num_inference_steps: int = 1,
        guidance_scale: float = 1.0,
        lora_scale: float = 1.0,
        dtype: str = "bf16",
        device_map: str = "balanced",
    ) -> None:
        from diffusers import WanImageToVideoPipeline

        self.height = int(height)
        self.view_width = int(view_width)
        self.num_views = int(num_views)
        self.num_frames = int(num_frames)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.lora_scale = float(lora_scale)
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            Path(base_model).as_posix(),
            torch_dtype=torch_dtype(dtype),
            device_map=device_map,
        )
        if lora_dir is not None and Path(lora_dir).as_posix():
            self.pipe.load_lora_weights(Path(lora_dir).as_posix(), weight_name="pytorch_lora_weights.safetensors")
        self.pipe.set_progress_bar_config(disable=True)

        call_params = set(inspect.signature(self.pipe.__call__).parameters)
        self._supports_last_image = "last_image" in call_params
        if not self._supports_last_image:
            raise RuntimeError(
                "The active diffusers WanImageToVideoPipeline does not accept last_image. "
                "This experiment requires FLF current+event-goal conditioning. Use the WAN/Finetrainers "
                "environment or a patched diffusers pipeline that supports last_image, or run with "
                "--wan-backend dummy for plumbing smoke tests."
            )

    @torch.no_grad()
    def __call__(
        self,
        current_images: Mapping[str, np.ndarray],
        goal_images: Mapping[str, np.ndarray],
        *,
        prompt: str,
        seed: int = 0,
    ) -> torch.Tensor:
        current = hstack_views(
            current_images,
            height=self.height,
            view_width=self.view_width,
            num_views=self.num_views,
        )
        goal = hstack_views(
            goal_images,
            height=self.height,
            view_width=self.view_width,
            num_views=self.num_views,
        )
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        output = self.pipe(
            image=current,
            last_image=goal,
            prompt=prompt,
            height=self.height,
            width=self.view_width * self.num_views,
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            output_type="latent",
            return_dict=True,
            attention_kwargs={"scale": float(self.lora_scale)},
        )
        return normalize_hstack_latents(as_tensor_from_pipeline_output(output), num_views=self.num_views)


def parse_latent_shape(value: str) -> tuple[int, int, int, int, int]:
    parts = tuple(int(x) for x in str(value).split(","))
    if len(parts) != 5:
        raise ValueError(f"Expected V,C,T,H,W latent shape, got {value!r}")
    return parts
