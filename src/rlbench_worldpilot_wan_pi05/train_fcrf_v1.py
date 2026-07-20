from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import logging
import os
from pathlib import Path
import random
import sys

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import tqdm

from .data import create_wan_latent_loader
from .latent_cache import load_latents, parse_latent_shape
from .modeling_fcrf import PI0FCRFV1Pytorch
from .train_torch import (
    build_config,
    cleanup_ddp,
    init_logging,
    init_wandb_if_needed,
    is_main_process,
    load_training_checkpoint,
    lr_at_step,
    move_batch_to_device,
    save_checkpoint,
    setup_ddp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FCRF-v1 on a frozen RLBench pi0.5-ft flow field.")
    parser.add_argument("config_name", nargs="?", default=os.environ.get("CONFIG_NAME", "pi05_rlbench_waypoint_h1"))
    parser.add_argument("--exp-name", default=os.environ.get("EXP_NAME", "selected10_fcrf_v1_pilot2k"))
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
    parser.add_argument(
        "--pi05-ft-weight-path",
        default=os.environ.get("PI05_FT_PYTORCH_WEIGHT_PATH"),
        help="Converted RLBench-finetuned pi0.5 checkpoint. A base-model checkpoint is not valid for FCRF-v1.",
    )
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
    parser.add_argument("--fcrf-num-heads", type=int, default=int(os.environ.get("FCRF_NUM_HEADS", "8")))
    parser.add_argument("--fcrf-dropout", type=float, default=float(os.environ.get("FCRF_DROPOUT", "0.0")))
    parser.add_argument("--fcrf-gate-bias", type=float, default=float(os.environ.get("FCRF_GATE_BIAS", "-2.2")))
    parser.add_argument("--residual-penalty", type=float, default=float(os.environ.get("FCRF_RESIDUAL_PENALTY", "1e-4")))
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
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-checkpoint", default=None)
    parser.add_argument("--num-eval-batches", type=int, default=50)
    parser.add_argument("--max-eval-samples", type=int, default=400)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--diagnostics-out", type=Path, default=None)
    return parser.parse_args()


def resolve_model_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        if path.name != "model.safetensors":
            raise ValueError(f"Expected model.safetensors or its parent directory, got {path}")
        return path
    model_file = path / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"Missing pi0.5-ft weights: {model_file}")
    return model_file


def load_pi05_ft_weights(model: PI0FCRFV1Pytorch, path: str | Path, device: torch.device) -> None:
    model_file = resolve_model_file(path)
    missing, unexpected = safetensors.torch.load_model(
        model,
        model_file,
        strict=False,
        device=str(device),
    )
    missing_base = sorted(name for name in missing if not name.startswith("fcrf."))
    if missing_base or unexpected:
        raise RuntimeError(
            "The supplied checkpoint is not a clean pi0.5-ft initialization: "
            f"missing_base={missing_base[:20]} unexpected={sorted(unexpected)[:20]}"
        )
    logging.info(
        "Loaded frozen pi0.5-ft from %s; fresh_fcrf_parameters=%d",
        model_file,
        len(missing),
    )


def trainable_parameter_summary(model: PI0FCRFV1Pytorch) -> dict[str, object]:
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen_names = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    return {
        "trainable_names": trainable_names,
        "trainable_tensors": len(trainable_names),
        "frozen_tensors": len(frozen_names),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "frozen_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
        ),
    }


def assert_fcrf_trainable_scope(model: PI0FCRFV1Pytorch) -> None:
    bad_trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad and not name.startswith("fcrf.")
    ]
    frozen_fcrf = [
        name for name, parameter in model.named_parameters() if name.startswith("fcrf.") and not parameter.requires_grad
    ]
    if bad_trainable or frozen_fcrf:
        raise RuntimeError(
            f"Invalid FCRF trainable scope: bad_trainable={bad_trainable} frozen_fcrf={frozen_fcrf}"
        )


def make_loader(args, config, local_batch_size: int, *, shuffle: bool):
    return create_wan_latent_loader(
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
        shuffle=shuffle,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
        allow_missing_latents=args.allow_missing_latents,
        expected_num_inference_steps=args.expected_wan_num_inference_steps,
        expected_backend=args.expected_wan_backend,
        expected_latent_shape=parse_latent_shape(args.expected_wan_latent_shape),
        rebuild_sample_index=args.rebuild_sample_index,
        skip_norm_stats=args.skip_norm_stats,
    )


def slice_batch(observation, actions, wan_latents, indices, count: int):
    observation = jax.tree.map(lambda value: value[:count], observation)
    return observation, actions[:count], wan_latents[:count], indices[:count]


class SameSkillShuffler:
    """Deterministically pair each row with another episode of the same skill."""

    def __init__(self, dataset, *, seed: int) -> None:
        self.dataset = dataset
        self.rows = {int(row["lerobot_index"]): row for row in dataset.sample_index}
        task_groups = defaultdict(list)
        event_groups = defaultdict(list)
        for row in dataset.sample_index:
            task_groups[str(row["task"])].append(row)
            event_groups[(str(row["task"]), int(row.get("event_idx", -1)))].append(row)

        rng = random.Random(int(seed))
        self.mapping = {}
        for index, row in self.rows.items():
            episode_key = (
                str(row.get("source_bundle")),
                str(row.get("variation")),
                str(row.get("episode")),
            )
            primary = event_groups[(str(row["task"]), int(row.get("event_idx", -1)))]
            candidates = [candidate for candidate in primary if self._episode_key(candidate) != episode_key]
            if not candidates:
                candidates = [
                    candidate
                    for candidate in task_groups[str(row["task"])]
                    if self._episode_key(candidate) != episode_key
                ]
            if not candidates:
                raise RuntimeError(f"No same-skill different-episode shuffle candidate for row {index}")
            self.mapping[index] = candidates[rng.randrange(len(candidates))]

    @staticmethod
    def _episode_key(row):
        return (
            str(row.get("source_bundle")),
            str(row.get("variation")),
            str(row.get("episode")),
        )

    def load(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        latents = []
        for raw_index in indices.detach().cpu().tolist():
            record = self.mapping[int(raw_index)]
            latent = load_latents(
                self.dataset.latent_cache_root,
                record,
                allow_missing=self.dataset.allow_missing_latents,
                dummy_shape=self.dataset.dummy_latent_shape,
                expected_num_inference_steps=self.dataset.expected_num_inference_steps,
                expected_backend=self.dataset.expected_backend,
                expected_shape=self.dataset.expected_latent_shape,
            )
            latents.append(latent)
        return torch.stack(latents, dim=0).to(device=device)

    def task_for(self, index: int) -> str:
        return str(self.rows[int(index)]["task"])


def _empty_metrics():
    return defaultdict(float)


def _update_metrics(metrics, base_outputs, matched, shuffled, sample_indices, shuffler):
    off_mse = (base_outputs["target_flow"] - base_outputs["base_flow"]).square().mean(dim=(1, 2))
    matched_mse = matched["flow_loss"].mean(dim=(1, 2))
    shuffled_mse = shuffled["flow_loss"].mean(dim=(1, 2))
    correction_norm = matched["correction"].flatten(1).norm(dim=1)
    residual_norm = matched["delta_flow"].flatten(1).norm(dim=1)
    gate = matched["gate"].flatten(1).mean(dim=1)
    cosine = matched["correction_cosine"]

    for position, raw_index in enumerate(sample_indices.detach().cpu().tolist()):
        values = {
            "n": 1.0,
            "off_mse": float(off_mse[position].detach().cpu()),
            "matched_mse": float(matched_mse[position].detach().cpu()),
            "shuffled_mse": float(shuffled_mse[position].detach().cpu()),
            "correction_cosine": float(cosine[position].detach().cpu()),
            "positive_cosine": float(cosine[position] > 0),
            "gate": float(gate[position].detach().cpu()),
            "correction_norm": float(correction_norm[position].detach().cpu()),
            "residual_norm": float(residual_norm[position].detach().cpu()),
        }
        for key, value in values.items():
            metrics["overall"][key] += value
            metrics["by_task"][shuffler.task_for(int(raw_index))][key] += value


def _finalize_metrics(raw):
    def finalize_group(values):
        count = int(values["n"])
        result = {"num_samples": count}
        for key, value in values.items():
            if key != "n":
                result[key] = float(value / max(1, count))
        result["matched_improvement_over_off"] = result["off_mse"] - result["matched_mse"]
        result["matched_improvement_over_shuffled"] = result["shuffled_mse"] - result["matched_mse"]
        return result

    return {
        "overall": finalize_group(raw["overall"]),
        "by_task": {task: finalize_group(values) for task, values in sorted(raw["by_task"].items())},
    }


@torch.no_grad()
def run_flow_diagnostics(model, loader, args, device, global_step: int) -> dict[str, object]:
    model.eval()
    shuffler = SameSkillShuffler(loader.dataset, seed=args.shuffle_seed)
    raw = {"overall": _empty_metrics(), "by_task": defaultdict(_empty_metrics)}
    processed = 0
    for batch_idx, (observation, actions, wan_latents, indices) in enumerate(loader):
        if batch_idx >= int(args.num_eval_batches) or processed >= int(args.max_eval_samples):
            break
        remaining = int(args.max_eval_samples) - processed
        if actions.shape[0] > remaining:
            observation, actions, wan_latents, indices = slice_batch(
                observation,
                actions,
                wan_latents,
                indices,
                remaining,
            )
        observation, actions, wan_latents = move_batch_to_device(
            observation,
            actions,
            wan_latents,
            device,
        )
        noise = model.sample_noise(actions.shape, device)
        time = model.sample_time(actions.shape[0], device)
        base_outputs = model.compute_base_outputs(
            observation,
            actions,
            noise=noise,
            time=time,
            preprocess_train=False,
        )
        matched = model.apply_fcrf(base_outputs, wan_latents, enabled=True)
        shuffled_latents = shuffler.load(indices, device)
        shuffled = model.apply_fcrf(base_outputs, shuffled_latents, enabled=True)
        _update_metrics(raw, base_outputs, matched, shuffled, indices, shuffler)
        processed += int(actions.shape[0])

    summary = {
        "method": "FCRF-v1",
        "checkpoint_step": int(global_step),
        "split": args.split,
        "shuffle": "same-task, preferably same-event-index, different episode",
        "shuffle_seed": int(args.shuffle_seed),
        **_finalize_metrics(raw),
    }
    if args.diagnostics_out is not None:
        args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return summary


def run_smoke(model, loader, args, device) -> None:
    model.train()
    observation, actions, wan_latents, _indices = next(iter(loader))
    observation, actions, wan_latents = move_batch_to_device(observation, actions, wan_latents, device)
    noise = model.sample_noise(actions.shape, device)
    time = model.sample_time(actions.shape[0], device)
    base_outputs = model.compute_base_outputs(observation, actions, noise=noise, time=time)
    initial = model.apply_fcrf(base_outputs, wan_latents, enabled=True)
    max_initial_difference = float((initial["final_flow"] - base_outputs["base_flow"]).abs().max().cpu())
    if max_initial_difference != 0.0:
        raise RuntimeError(f"FCRF initial identity failed: max_abs_diff={max_initial_difference}")

    loss = initial["flow_loss"].mean() + float(args.residual_penalty) * initial["residual_penalty"].mean()
    loss.backward()
    base_grads = [
        name
        for name, parameter in model.named_parameters()
        if not name.startswith("fcrf.") and parameter.grad is not None
    ]
    fcrf_grad_tensors = sum(
        int(parameter.grad is not None)
        for name, parameter in model.named_parameters()
        if name.startswith("fcrf.")
    )
    fcrf_nonzero_grad_tensors = sum(
        int(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)))
        for name, parameter in model.named_parameters()
        if name.startswith("fcrf.")
    )
    if base_grads or fcrf_grad_tensors == 0 or fcrf_nonzero_grad_tensors == 0:
        raise RuntimeError(
            "FCRF gradient smoke failed: "
            f"base_grads={base_grads} fcrf_grad_tensors={fcrf_grad_tensors} "
            f"fcrf_nonzero_grad_tensors={fcrf_nonzero_grad_tensors}"
        )
    print(
        json.dumps(
            {
                "smoke": "passed",
                "initial_max_abs_flow_difference": max_initial_difference,
                "initial_gate_mean": float(initial["gate"].mean().detach().cpu()),
                "fcrf_grad_tensors": int(fcrf_grad_tensors),
                "fcrf_nonzero_grad_tensors": int(fcrf_nonzero_grad_tensors),
                "base_grad_tensors": len(base_grads),
                **trainable_parameter_summary(model),
            },
            sort_keys=True,
        )
    )


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
    loader, data_config = make_loader(
        args,
        config,
        local_batch_size,
        shuffle=not (args.eval_only or args.smoke_only or args.dry_run),
    )

    if args.dry_run:
        observation, actions, wan_latents, indices = next(iter(loader))
        del observation
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "method": "FCRF-v1",
                    "config": config.name,
                    "exp_name": config.exp_name,
                    "actions_shape": list(actions.shape),
                    "wan_latents_shape": list(wan_latents.shape),
                    "lerobot_index_head": [int(value) for value in indices[:4]],
                },
                sort_keys=True,
            )
        )
        cleanup_ddp()
        return

    if args.pi05_ft_weight_path is None:
        raise ValueError(
            "--pi05-ft-weight-path / PI05_FT_PYTORCH_WEIGHT_PATH is required. "
            "It must be the RLBench-finetuned pi0.5 checkpoint, not pi0.5 base."
        )

    model = PI0FCRFV1Pytorch(
        config.model,
        fcrf_num_heads=args.fcrf_num_heads,
        fcrf_dropout=args.fcrf_dropout,
        fcrf_gate_bias=args.fcrf_gate_bias,
    ).to(device)
    first_latents = torch.as_tensor(loader.dataset[0]["wan_latents"])[None].to(device)
    init_dtype = torch.bfloat16 if config.pytorch_training_precision == "bfloat16" else torch.float32
    model.initialize_fcrf(first_latents, device=device, dtype=init_dtype)
    load_pi05_ft_weights(model, args.pi05_ft_weight_path, device)
    model.freeze_base()
    assert_fcrf_trainable_scope(model)
    if is_main:
        logging.info("FCRF scope: %s", json.dumps(trainable_parameter_summary(model), sort_keys=True))

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.lr_schedule.peak_lr),
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
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
        global_step = load_training_checkpoint(
            model,
            optimizer if args.resume and not args.eval_only else None,
            checkpoint_dir,
            device,
            step=step,
        )
        if is_main:
            logging.info("Loaded FCRF checkpoint at step %d", global_step)

    if args.smoke_only:
        if use_ddp:
            raise ValueError("Run --smoke-only with NPROC_PER_NODE=1")
        run_smoke(model, loader, args, device)
        cleanup_ddp()
        return

    if args.eval_only:
        if use_ddp:
            raise ValueError("Run --eval-only with NPROC_PER_NODE=1")
        if not args.eval_checkpoint:
            raise ValueError("--eval-only requires --eval-checkpoint")
        run_flow_diagnostics(model, loader, args, device, global_step)
        cleanup_ddp()
        return

    if config.checkpoint_dir.exists() and global_step == 0:
        if args.overwrite:
            import shutil

            shutil.rmtree(config.checkpoint_dir)
        else:
            raise FileExistsError(f"{config.checkpoint_dir} exists; pass --overwrite or --resume")
    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    model.train()
    wandb = init_wandb_if_needed(config, args, config.wandb_enabled)
    pbar = tqdm.tqdm(
        total=int(config.num_train_steps),
        initial=global_step,
        disable=not is_main,
        desc="FCRF-v1",
    )
    while global_step < int(config.num_train_steps):
        if use_ddp and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(global_step)
        for observation, actions, wan_latents, _indices in loader:
            if global_step >= int(config.num_train_steps):
                break
            observation, actions, wan_latents = move_batch_to_device(
                observation,
                actions,
                wan_latents,
                device,
            )
            lr = lr_at_step(config, global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            outputs = model(observation, actions, wan_latents=wan_latents)
            flow_loss = outputs["flow_loss"].mean()
            residual_penalty = outputs["residual_penalty"].mean()
            loss = flow_loss + float(args.residual_penalty) * residual_penalty
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=config.optimizer.clip_gradient_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if is_main and global_step % int(config.log_interval) == 0:
                base_mse = (outputs["target_flow"] - outputs["base_flow"]).square().mean()
                metrics = {
                    "train/loss": float(loss.detach().cpu()),
                    "train/flow_mse": float(flow_loss.detach().cpu()),
                    "train/base_mse": float(base_mse.detach().cpu()),
                    "train/residual_penalty": float(residual_penalty.detach().cpu()),
                    "train/correction_cosine": float(outputs["correction_cosine"].mean().detach().cpu()),
                    "train/gate_mean": float(outputs["gate"].mean().detach().cpu()),
                    "train/correction_norm": float(outputs["correction"].flatten(1).norm(dim=1).mean().detach().cpu()),
                    "train/lr": float(lr),
                    "train/grad_norm": float(grad_norm),
                }
                logging.info("step=%d %s", global_step, json.dumps(metrics, sort_keys=True))
                if wandb is not None:
                    wandb.log(metrics, step=global_step)

            global_step += 1
            if is_main:
                pbar.update(1)
            if is_main and (
                global_step % int(config.save_interval) == 0
                or global_step == int(config.num_train_steps)
            ):
                save_checkpoint(model, optimizer, global_step, config, data_config, args)
                logging.info("Saved FCRF-v1 checkpoint at step %d", global_step)

            del observation, actions, wan_latents, outputs, loss, flow_loss, residual_penalty
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
