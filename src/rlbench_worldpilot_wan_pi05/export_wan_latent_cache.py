from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from .latent_cache import latent_path_for_record, save_latent_record
from .sample_index import VIEW_NAMES, build_sample_index, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export WAN VAE-before-decode future-video latent cache.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--event-manifest-path", type=Path, default=None)
    parser.add_argument("--goal-mode", choices=("event_end", "next_waypoint"), default="event_end")
    parser.add_argument("--sample-index-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test", "all"))
    parser.add_argument("--sample-every-n", type=int, default=0)
    parser.add_argument("--rgb-root-200", type=Path, required=True)
    parser.add_argument("--rgb-root-400", type=Path, required=True)
    parser.add_argument("--backend", choices=("dummy", "wan-diffusers"), default="dummy")
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--lora-dir", type=Path, default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--view-width", type=int, default=256)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--num-frames", type=int, default=21)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dummy-shape", default="3,16,5,28,28")
    parser.add_argument("--dummy-fill", choices=("zeros", "randn"), default="zeros")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def parse_shape(value: str) -> tuple[int, int, int, int, int]:
    parts = tuple(int(x) for x in value.split(","))
    if len(parts) != 5:
        raise ValueError("--dummy-shape must be V,C,T,H,W")
    return parts


def ensure_sample_index(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_index_path.exists() and not args.overwrite:
        rows = read_jsonl(args.sample_index_path)
    else:
        rows = build_sample_index(
            args.manifest_path,
            event_manifest_path=args.event_manifest_path,
            goal_mode=args.goal_mode,
            split=args.split,
            sample_every_n=args.sample_every_n,
            rgb_root_200=args.rgb_root_200,
            rgb_root_400=args.rgb_root_400,
        )
        write_jsonl(args.sample_index_path, rows)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if args.num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    return rows


def load_hstack(paths_by_view: dict[str, str], *, height: int, view_width: int, num_views: int) -> Image.Image:
    canvas = Image.new("RGB", (view_width * num_views, height))
    for view_idx, view in enumerate(VIEW_NAMES[:num_views]):
        image = Image.open(paths_by_view[view]).convert("RGB").resize((view_width, height), Image.BICUBIC)
        canvas.paste(image, (view_idx * view_width, 0))
    return canvas


def as_tensor_from_pipeline_output(output) -> torch.Tensor:
    value = getattr(output, "frames", output)
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Pipeline returned an empty latent output")
        value = value[0]
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError(f"Unsupported WAN latent output type: {type(value)!r}")


def normalize_hstack_latents(latents: torch.Tensor, *, num_views: int) -> torch.Tensor:
    latents = latents.detach().cpu()
    if latents.ndim != 5:
        raise ValueError(f"Expected WAN latent tensor with 5 dims, got {tuple(latents.shape)}")
    # Common diffusers shape: [B,C,T,H,W]. Some pipelines return [B,T,C,H,W].
    if latents.shape[1] <= 64:
        bcthw = latents
    else:
        bcthw = latents.permute(0, 2, 1, 3, 4).contiguous()
    if bcthw.shape[0] != 1:
        raise ValueError(f"Expected batch size 1 from WAN pipeline, got {tuple(bcthw.shape)}")
    c, t, h, w_total = bcthw.shape[1:]
    if w_total % num_views != 0:
        raise ValueError(f"Latent width {w_total} is not divisible by num_views={num_views}")
    w_view = w_total // num_views
    return bcthw[0].reshape(c, t, h, num_views, w_view).permute(3, 0, 1, 2, 4).contiguous()


class WanDiffusersBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.base_model is None:
            raise ValueError("--base-model is required for --backend wan-diffusers")
        from diffusers import WanImageToVideoPipeline

        self.args = args
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            args.base_model.as_posix(),
            torch_dtype=torch_dtype(args.dtype),
            device_map=args.device_map,
        )
        if args.lora_dir is not None and args.lora_dir.as_posix():
            self.pipe.load_lora_weights(args.lora_dir.as_posix(), weight_name="pytorch_lora_weights.safetensors")
        self.pipe.set_progress_bar_config(disable=True)
        if "last_image" not in set(inspect.signature(self.pipe.__call__).parameters):
            raise RuntimeError(
                "The active diffusers WanImageToVideoPipeline does not accept last_image. "
                "This experiment requires FLF current+event-goal conditioning. Use the WAN/Finetrainers "
                "environment or a patched diffusers pipeline that supports last_image, or run with "
                "--backend dummy for plumbing smoke tests."
            )

    def __call__(self, record: dict[str, Any]) -> torch.Tensor:
        current = load_hstack(
            record["current_image_paths"],
            height=self.args.height,
            view_width=self.args.view_width,
            num_views=self.args.num_views,
        )
        goal_paths = record.get("latent_goal_image_paths") or record["target_image_paths"]
        target = load_hstack(
            goal_paths,
            height=self.args.height,
            view_width=self.args.view_width,
            num_views=self.args.num_views,
        )
        prompt = str(record.get("task_instruction") or record.get("task") or "")
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(self.args.seed + int(record["lerobot_index"]))
        with torch.no_grad():
            output = self.pipe(
                image=current,
                last_image=target,
                prompt=prompt,
                height=self.args.height,
                width=self.args.view_width * self.args.num_views,
                num_frames=self.args.num_frames,
                num_inference_steps=self.args.num_inference_steps,
                guidance_scale=self.args.guidance_scale,
                generator=generator,
                output_type="latent",
                return_dict=True,
                attention_kwargs={"scale": float(self.args.lora_scale)},
            )
        return normalize_hstack_latents(as_tensor_from_pipeline_output(output), num_views=self.args.num_views)


def dummy_latents(args: argparse.Namespace, record: dict[str, Any]) -> torch.Tensor:
    shape = parse_shape(args.dummy_shape)
    if args.dummy_fill == "zeros":
        return torch.zeros(shape, dtype=torch.float16)
    generator = torch.Generator().manual_seed(args.seed + int(record["lerobot_index"]))
    return torch.randn(shape, generator=generator, dtype=torch.float16)


def main() -> None:
    args = parse_args()
    rows = ensure_sample_index(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    backend = WanDiffusersBackend(args) if args.backend == "wan-diffusers" else None

    written = 0
    skipped = 0
    for record in tqdm(rows, desc=f"Export {args.backend} WAN latents"):
        out_path = latent_path_for_record(args.out_dir, record)
        if out_path.exists() and args.resume and not args.overwrite:
            skipped += 1
            continue
        latents = backend(record) if backend is not None else dummy_latents(args, record)
        save_latent_record(
            out_path,
            latents,
            record=record,
            latent_layout="vcthw",
            backend=args.backend,
            metadata={
                "num_inference_steps": int(args.num_inference_steps),
                "num_frames": int(args.num_frames),
                "guidance_scale": float(args.guidance_scale),
                "lora_scale": float(args.lora_scale),
            },
        )
        written += 1

    summary = {
        "backend": args.backend,
        "out_dir": args.out_dir.as_posix(),
        "sample_index_path": args.sample_index_path.as_posix(),
        "num_rows_selected": len(rows),
        "written": written,
        "skipped": skipped,
        "split": args.split,
        "goal_mode": args.goal_mode,
        "event_manifest_path": args.event_manifest_path.as_posix() if args.event_manifest_path else None,
        "num_inference_steps": int(args.num_inference_steps),
        "num_frames": int(args.num_frames),
    }
    (args.out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
