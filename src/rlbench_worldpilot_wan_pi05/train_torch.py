from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import tqdm

import openpi.models.pi0_config
import openpi.shared.normalize as _normalize
import openpi.training.config as _config

from .data import create_wan_latent_loader
from .latent_cache import parse_latent_shape
from .modeling import PI0WanLatentSteeringPytorch


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_ddp() -> tuple[bool, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pi0.5 PyTorch with WorldPilot-style WAN latent steering.")
    parser.add_argument("config_name", nargs="?", default=os.environ.get("CONFIG_NAME", "pi05_rlbench_waypoint_h1"))
    parser.add_argument("--exp-name", default=os.environ.get("EXP_NAME", "selected10_worldpilot_wan_pi05_torch"))
    parser.add_argument("--lerobot-repo-id", default=os.environ.get("LEROBOT_REPO_ID", "rlbench/selected10_pi05_waypoint_h1"))
    parser.add_argument("--manifest-path", default=os.environ.get("MANIFEST_PATH"), required=os.environ.get("MANIFEST_PATH") is None)
    parser.add_argument("--event-manifest-path", default=os.environ.get("EVENT_MANIFEST_PATH"))
    parser.add_argument("--goal-mode", choices=("event_end", "next_waypoint"), default=os.environ.get("WAN_LATENT_GOAL_MODE", "event_end"))
    parser.add_argument("--sample-index-path", default=os.environ.get("SAMPLE_INDEX_PATH"))
    parser.add_argument("--wan-latent-cache-root", default=os.environ.get("WAN_LATENT_CACHE_ROOT"), required=os.environ.get("WAN_LATENT_CACHE_ROOT") is None)
    parser.add_argument("--split", default=os.environ.get("SPLIT", "train"), choices=("train", "val", "test", "all"))
    parser.add_argument("--sample-every-n", type=int, default=int(os.environ.get("SAMPLE_EVERY_N", "0")))
    parser.add_argument("--rgb-root-200", default=os.environ.get("RGB_ROOT_200"))
    parser.add_argument("--rgb-root-400", default=os.environ.get("RGB_ROOT_400"))
    parser.add_argument("--pytorch-weight-path", default=os.environ.get("PYTORCH_WEIGHT_PATH"))
    parser.add_argument("--checkpoint-base-dir", default=os.environ.get("CHECKPOINT_BASE_DIR", "./checkpoints"))
    parser.add_argument("--assets-base-dir", default=os.environ.get("ASSETS_BASE_DIR", "./assets"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--keep-period", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--lr-schedule.warmup-steps", dest="warmup_steps", type=int, default=None)
    parser.add_argument("--lr-schedule.peak-lr", dest="peak_lr", type=float, default=None)
    parser.add_argument("--lr-schedule.decay-steps", dest="decay_steps", type=int, default=None)
    parser.add_argument("--lr-schedule.decay-lr", dest="decay_lr", type=float, default=None)
    parser.add_argument("--pytorch-training-precision", choices=("bfloat16", "float32"), default=os.environ.get("PYTORCH_TRAINING_PRECISION"))
    parser.add_argument("--wan-num-heads", type=int, default=int(os.environ.get("WAN_FUSER_NUM_HEADS", "8")))
    parser.add_argument("--wan-dropout", type=float, default=float(os.environ.get("WAN_FUSER_DROPOUT", "0.0")))
    parser.add_argument("--wan-steering-mode", choices=("early", "block"), default=os.environ.get("WAN_STEERING_MODE", "early"))
    parser.add_argument("--wan-steering-block", type=int, default=int(os.environ.get("WAN_STEERING_BLOCK", "12")))
    parser.add_argument("--wan-steering-gate", choices=("auto", "on", "off"), default=os.environ.get("WAN_STEERING_GATE", "auto"))
    parser.add_argument(
        "--trainable-scope",
        choices=("all", "wan_fuser", "wan_fuser_action_head"),
        default=os.environ.get("TRAINABLE_SCOPE", "all"),
        help=(
            "Which parameters to optimize. Use wan_fuser on 40GB GPUs to keep pi0.5 frozen and train only "
            "the WorldPilot-style latent steering adapter."
        ),
    )
    parser.add_argument("--expected-wan-num-inference-steps", type=int, default=None)
    parser.add_argument("--expected-wan-backend", default=os.environ.get("WAN_EXPECTED_BACKEND"))
    parser.add_argument("--expected-wan-latent-shape", default=os.environ.get("WAN_LATENT_SHAPE", "3,16,6,32,32"))
    parser.add_argument("--overwrite", action="store_true", default=os.environ.get("OVERWRITE", "0") == "1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb-enabled", dest="wandb_enabled", action="store_false")
    parser.add_argument("--wandb-enabled", dest="wandb_enabled", action="store_true")
    parser.set_defaults(wandb_enabled=os.environ.get("WANDB_ENABLED", "1") != "0")
    parser.add_argument("--allow-missing-latents", action="store_true")
    parser.add_argument("--rebuild-sample-index", action="store_true")
    parser.add_argument("--skip-norm-stats", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dry-run-model",
        action="store_true",
        help="Also initialize the model, load weights, configure trainable params, and build the optimizer, then exit.",
    )
    parser.add_argument(
        "--dry-run-step",
        action="store_true",
        help="Run one forward/backward/optimizer step, report memory, and exit without writing a checkpoint.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-checkpoint", default=None)
    parser.add_argument("--num-eval-batches", type=int, default=50)
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    config = _config.get_config(args.config_name)
    data_factory = dataclasses.replace(config.data, repo_id=args.lerobot_repo_id)
    config = dataclasses.replace(
        config,
        exp_name=args.exp_name,
        data=data_factory,
        checkpoint_base_dir=args.checkpoint_base_dir,
        assets_base_dir=args.assets_base_dir,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb_enabled,
    )
    replace_kwargs = {}
    for field in ("batch_size", "num_train_steps", "num_workers", "save_interval", "keep_period", "log_interval"):
        value = getattr(args, field)
        if value is not None:
            replace_kwargs[field] = value
    if args.pytorch_training_precision is not None:
        replace_kwargs["pytorch_training_precision"] = args.pytorch_training_precision
    if replace_kwargs:
        config = dataclasses.replace(config, **replace_kwargs)

    lr_kwargs = {}
    for arg_name, field_name in (
        ("warmup_steps", "warmup_steps"),
        ("peak_lr", "peak_lr"),
        ("decay_steps", "decay_steps"),
        ("decay_lr", "decay_lr"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            lr_kwargs[field_name] = value
    if lr_kwargs:
        config = dataclasses.replace(config, lr_schedule=dataclasses.replace(config.lr_schedule, **lr_kwargs))
    object.__setattr__(config.model, "dtype", config.pytorch_training_precision)
    return config


def save_checkpoint(model, optimizer, global_step: int, config, data_config, args) -> None:
    ckpt_dir = config.checkpoint_dir / f"{global_step}"
    tmp_dir = config.checkpoint_dir / f"tmp_{global_step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.save_model(model_to_save, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(
        {
            "global_step": int(global_step),
            "config_name": config.name,
            "args": vars(args),
            "timestamp": time.time(),
        },
        tmp_dir / "metadata.pt",
    )
    if data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_dir / "assets" / data_config.asset_id, data_config.norm_stats)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    tmp_dir.rename(ckpt_dir)

    if config.keep_period is not None:
        for child in config.checkpoint_dir.iterdir():
            if not child.is_dir() or not child.name.isdigit():
                continue
            step = int(child.name)
            if step == global_step or step % int(config.keep_period) == 0:
                continue
            if step < global_step and step % int(config.save_interval) == 0:
                shutil.rmtree(child)


def latest_checkpoint_step(checkpoint_dir: Path) -> int:
    steps = [int(path.name) for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    if not steps:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")
    return max(steps)


def load_training_checkpoint(model, optimizer, checkpoint_dir: Path, device: torch.device, step: int | None = None) -> int:
    step = latest_checkpoint_step(checkpoint_dir) if step is None else int(step)
    ckpt_dir = checkpoint_dir / str(step)
    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.load_model(model_to_load, ckpt_dir / "model.safetensors", strict=True, device=str(device))
    optim_path = ckpt_dir / "optimizer.pt"
    if optimizer is not None and optim_path.exists():
        optimizer.load_state_dict(torch.load(optim_path, map_location=device, weights_only=False))
    metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
    return int(metadata.get("global_step", step))


def lr_at_step(config, step: int) -> float:
    warmup_steps = int(config.lr_schedule.warmup_steps)
    peak_lr = float(config.lr_schedule.peak_lr)
    decay_steps = int(config.lr_schedule.decay_steps)
    end_lr = float(config.lr_schedule.decay_lr)
    if step < warmup_steps:
        init_lr = peak_lr / (warmup_steps + 1)
        return init_lr + (peak_lr - init_lr) * step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
    return end_lr + (peak_lr - end_lr) * 0.5 * (1 + np.cos(np.pi * progress))


def move_batch_to_device(observation, actions, wan_latents, device):
    observation = jax.tree.map(lambda x: x.to(device), observation)
    actions = actions.to(device=device, dtype=torch.float32)
    wan_latents = wan_latents.to(device=device)
    return observation, actions, wan_latents


def init_wandb_if_needed(config, args, enabled: bool):
    if not enabled or not is_main_process():
        return None
    try:
        import wandb
    except Exception:
        logging.warning("wandb is not installed; continuing without wandb")
        return None
    wandb.init(project=config.project_name, name=config.exp_name, config=vars(args))
    return wandb


def configure_trainable_parameters(model: torch.nn.Module, scope: str) -> tuple[int, int]:
    if scope == "all":
        for param in model.parameters():
            param.requires_grad_(True)
    elif scope == "wan_fuser":
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.wan_fuser.parameters():
            param.requires_grad_(True)
    elif scope == "wan_fuser_action_head":
        for param in model.parameters():
            param.requires_grad_(False)
        for module in (model.wan_fuser, model.action_out_proj):
            for param in module.parameters():
                param.requires_grad_(True)
    else:
        raise ValueError(f"Unsupported trainable scope: {scope}")

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable == 0:
        raise ValueError(f"No trainable parameters for trainable scope {scope!r}")
    return total, trainable


def main() -> None:
    init_logging()
    args = parse_args()
    config = build_config(args)
    if args.sample_index_path is None:
        args.sample_index_path = str(Path(args.wan_latent_cache_root) / f"sample_index_{args.split}.jsonl")

    use_ddp, local_rank, device = setup_ddp()
    is_main = is_main_process()
    torch.manual_seed(int(config.seed) + local_rank)
    np.random.seed(int(config.seed) + local_rank)

    world_size = dist.get_world_size() if use_ddp else 1
    if config.batch_size % world_size != 0:
        raise ValueError(f"batch_size={config.batch_size} must be divisible by world_size={world_size}")
    local_batch_size = config.batch_size // world_size

    if is_main:
        logging.info("Config: %s exp=%s world_size=%d local_batch=%d", config.name, config.exp_name, world_size, local_batch_size)
        logging.info("WAN latent cache: %s", args.wan_latent_cache_root)
        logging.info("WAN latent goal_mode: %s event_manifest=%s", args.goal_mode, args.event_manifest_path)
        logging.info("Sample index: %s", args.sample_index_path)

    loader, data_config = create_wan_latent_loader(
        config,
        manifest_path=args.manifest_path,
        event_manifest_path=args.event_manifest_path,
        goal_mode=args.goal_mode,
        sample_index_path=args.sample_index_path,
        latent_cache_root=args.wan_latent_cache_root,
        split=args.split,
        sample_every_n=args.sample_every_n,
        rgb_root_200=args.rgb_root_200,
        rgb_root_400=args.rgb_root_400,
        local_batch_size=local_batch_size,
        shuffle=not args.eval_only,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
        allow_missing_latents=args.allow_missing_latents,
        expected_num_inference_steps=args.expected_wan_num_inference_steps,
        expected_backend=args.expected_wan_backend,
        expected_latent_shape=parse_latent_shape(args.expected_wan_latent_shape),
        rebuild_sample_index=args.rebuild_sample_index,
        skip_norm_stats=args.skip_norm_stats,
    )

    if args.dry_run:
        batch = next(iter(loader))
        observation, actions, wan_latents, lerobot_index = batch
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "config": config.name,
                    "exp_name": config.exp_name,
                    "actions_shape": list(actions.shape),
                    "wan_latents_shape": list(wan_latents.shape),
                    "wan_steering_mode": args.wan_steering_mode,
                    "wan_steering_block": int(args.wan_steering_block),
                    "wan_steering_gate": args.wan_steering_gate,
                    "lerobot_index_head": [int(x) for x in lerobot_index[: min(4, len(lerobot_index))]],
                },
                sort_keys=True,
            )
        )
        cleanup_ddp()
        return

    model = PI0WanLatentSteeringPytorch(
        config.model,
        wan_num_heads=args.wan_num_heads,
        wan_dropout=args.wan_dropout,
        wan_steering_mode=args.wan_steering_mode,
        wan_steering_block=args.wan_steering_block,
        wan_steering_gate=args.wan_steering_gate,
    ).to(device)

    first_latents = torch.as_tensor(loader.dataset[0]["wan_latents"])[None].to(device)
    init_dtype = torch.bfloat16 if config.pytorch_training_precision == "bfloat16" else torch.float32
    model.initialize_wan_fuser(first_latents, device=device, dtype=init_dtype)

    if args.pytorch_weight_path:
        model_path = Path(args.pytorch_weight_path) / "model.safetensors"
        missing, unexpected = safetensors.torch.load_model(model, model_path, strict=False, device=str(device))
        if is_main:
            logging.info("Loaded base PyTorch weights from %s; missing=%d unexpected=%d", model_path, len(missing), len(unexpected))

    total_params, trainable_params = configure_trainable_parameters(model, args.trainable_scope)
    if is_main:
        logging.info(
            "Trainable scope=%s trainable_params=%d total_params=%d trainable_ratio=%.6f",
            args.trainable_scope,
            trainable_params,
            total_params,
            trainable_params / max(total_params, 1),
        )

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.lr_schedule.peak_lr),
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    if args.dry_run_model:
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            max_memory_allocated = torch.cuda.max_memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
        else:
            max_memory_allocated = 0
            memory_reserved = 0
        if is_main:
            print(
                json.dumps(
                    {
                        "dry_run_model": True,
                        "config": config.name,
                        "exp_name": config.exp_name,
                        "trainable_scope": args.trainable_scope,
                        "trainable_params": trainable_params,
                        "total_params": total_params,
                        "trainable_ratio": trainable_params / max(total_params, 1),
                        "optimizer_param_groups": len(optimizer.param_groups),
                        "optimizer_params": sum(param.numel() for group in optimizer.param_groups for param in group["params"]),
                        "cuda_max_memory_allocated_gb": max_memory_allocated / (1024**3),
                        "cuda_memory_reserved_gb": memory_reserved / (1024**3),
                    },
                    sort_keys=True,
                )
            )
        cleanup_ddp()
        return

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    global_step = 0
    if args.resume or args.eval_checkpoint:
        if args.eval_checkpoint:
            ckpt_path = Path(args.eval_checkpoint)
            step = int(ckpt_path.name) if ckpt_path.name.isdigit() else None
            checkpoint_dir = ckpt_path.parent if step is not None else ckpt_path
        else:
            checkpoint_dir = config.checkpoint_dir
            step = None
        global_step = load_training_checkpoint(model, optimizer if not args.eval_only else None, checkpoint_dir, device, step=step)
        if is_main:
            logging.info("Loaded checkpoint at step %d", global_step)
    elif config.checkpoint_dir.exists():
        if args.overwrite:
            shutil.rmtree(config.checkpoint_dir)
        else:
            raise FileExistsError(f"{config.checkpoint_dir} exists; pass --overwrite or --resume")
    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wandb = init_wandb_if_needed(config, args, config.wandb_enabled and not args.eval_only)

    if args.eval_only:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch_idx, (observation, actions, wan_latents, _indices) in enumerate(loader):
                if batch_idx >= args.num_eval_batches:
                    break
                observation, actions, wan_latents = move_batch_to_device(observation, actions, wan_latents, device)
                loss = model(observation, actions, wan_latents=wan_latents).mean()
                losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        if is_main:
            print(json.dumps({"eval_only": True, "global_step": global_step, "mean_loss": mean_loss, "num_batches": len(losses)}))
        cleanup_ddp()
        return

    if args.dry_run_step:
        model.train()
        observation, actions, wan_latents, _indices = next(iter(loader))
        observation, actions, wan_latents = move_batch_to_device(observation, actions, wan_latents, device)
        losses = model(observation, actions, wan_latents=wan_latents)
        loss = losses.mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=config.optimizer.clip_gradient_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            max_memory_allocated = torch.cuda.max_memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
        else:
            max_memory_allocated = 0
            memory_reserved = 0
        if is_main:
            print(
                json.dumps(
                    {
                        "dry_run_step": True,
                        "config": config.name,
                        "exp_name": config.exp_name,
                        "trainable_scope": args.trainable_scope,
                        "loss": float(loss.detach().cpu()),
                        "grad_norm": float(grad_norm),
                        "trainable_params": trainable_params,
                        "total_params": total_params,
                        "cuda_max_memory_allocated_gb": max_memory_allocated / (1024**3),
                        "cuda_memory_reserved_gb": memory_reserved / (1024**3),
                    },
                    sort_keys=True,
                )
            )
        cleanup_ddp()
        return

    model.train()
    pbar = tqdm.tqdm(total=int(config.num_train_steps), initial=global_step, disable=not is_main, desc="Training")
    while global_step < int(config.num_train_steps):
        if use_ddp and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(global_step)
        for observation, actions, wan_latents, _indices in loader:
            if global_step >= int(config.num_train_steps):
                break
            observation, actions, wan_latents = move_batch_to_device(observation, actions, wan_latents, device)
            lr = lr_at_step(config, global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            losses = model(observation, actions, wan_latents=wan_latents)
            loss = losses.mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=config.optimizer.clip_gradient_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if is_main and global_step % int(config.log_interval) == 0:
                loss_value = float(loss.detach().cpu())
                logging.info("step=%d loss=%.6f lr=%.3e grad_norm=%.4f", global_step, loss_value, lr, float(grad_norm))
                if wandb is not None:
                    wandb.log({"train/loss": loss_value, "train/lr": lr, "train/grad_norm": float(grad_norm)}, step=global_step)

            global_step += 1
            if is_main:
                pbar.update(1)
            if is_main and (global_step % int(config.save_interval) == 0 or global_step == int(config.num_train_steps)):
                save_checkpoint(model, optimizer, global_step, config, data_config, args)
                logging.info("Saved checkpoint at step %d", global_step)

            del observation, actions, wan_latents, losses, loss
            if torch.cuda.is_available() and global_step < 5:
                torch.cuda.empty_cache()
            gc.collect()

    if wandb is not None:
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup_ddp()
        sys.exit(130)
