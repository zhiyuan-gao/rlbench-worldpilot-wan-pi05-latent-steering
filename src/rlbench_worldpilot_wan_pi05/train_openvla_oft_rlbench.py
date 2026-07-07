from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .latent_cache import parse_latent_shape
from .openvla_oft_data import (
    DATASET_STATS_KEY,
    OpenVLAOFTRLBenchDataset,
    WanLatentCollator,
    build_or_load_sample_index,
    fit_or_load_stats,
)
from .openvla_oft_steering import OpenVLAOFTWanActionHead, ensure_openvla_oft_on_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune OpenVLA-OFT on RLBench selected10 waypoint samples.")
    parser.add_argument("--openvla-oft-dir", type=Path, default=Path(os.environ.get("OPENVLA_OFT_DIR", "")))
    parser.add_argument("--vla-path", default=os.environ.get("OPENVLA_OFT_VLA_PATH", "openvla/openvla-7b"))
    parser.add_argument("--exp-name", default=os.environ.get("EXP_NAME", "rlbench_openvla_oft_waypoint"))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("OPENVLA_OFT_RUN_ROOT", "checkpoints/openvla_oft_rlbench")))
    parser.add_argument("--resume-from", type=Path, default=None)

    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--event-manifest-path", type=Path, default=None)
    parser.add_argument("--sample-index-path", type=Path, required=True)
    parser.add_argument("--goal-mode", choices=("event_end", "next_waypoint"), default=os.environ.get("WAN_LATENT_GOAL_MODE", "event_end"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default=os.environ.get("SPLIT", "train"))
    parser.add_argument("--sample-every-n", type=int, default=int(os.environ.get("WAN_SAMPLE_EVERY_N", "0")))
    parser.add_argument("--rebuild-sample-index", action="store_true")
    parser.add_argument("--rgb-root-200", type=Path, required=True)
    parser.add_argument("--rgb-root-400", type=Path, required=True)
    parser.add_argument("--lowdim-root-200", type=Path, required=True)
    parser.add_argument("--lowdim-root-400", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--refit-stats", action="store_true")

    parser.add_argument("--num-images-in-input", type=int, default=int(os.environ.get("OPENVLA_OFT_NUM_IMAGES_IN_INPUT", "3")))
    parser.add_argument("--num-actions-chunk", type=int, default=int(os.environ.get("OPENVLA_OFT_NUM_ACTIONS_CHUNK", "8")))
    parser.add_argument("--action-dim", type=int, default=int(os.environ.get("OPENVLA_OFT_ACTION_DIM", "7")))
    parser.add_argument("--proprio-dim", type=int, default=int(os.environ.get("OPENVLA_OFT_PROPRIO_DIM", "7")))
    parser.add_argument("--use-proprio", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("OPENVLA_OFT_LR", "5e-4")))
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("OPENVLA_OFT_BATCH_SIZE", "1")))
    parser.add_argument("--grad-accumulation-steps", type=int, default=int(os.environ.get("OPENVLA_OFT_GRAD_ACCUMULATION_STEPS", "1")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("OPENVLA_OFT_MAX_STEPS", "100000")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("OPENVLA_OFT_NUM_WORKERS", "4")))
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--use-wan-steering", action="store_true")
    parser.add_argument("--wan-latent-cache-root", type=Path, default=Path(os.environ.get("WAN_LATENT_CACHE_ROOT", "")))
    parser.add_argument("--expected-wan-num-inference-steps", type=int, default=None)
    parser.add_argument("--expected-wan-backend", default=os.environ.get("WAN_EXPECTED_BACKEND"))
    parser.add_argument("--expected-wan-latent-shape", default=os.environ.get("WAN_LATENT_SHAPE"))
    parser.add_argument("--allow-missing-wan-latents", action="store_true")
    parser.add_argument("--wan-fuser-num-heads", type=int, default=8)
    parser.add_argument("--wan-fuser-dropout", type=float, default=0.0)

    parser.add_argument("--dry-run", action="store_true", help="Validate RLBench sample index/stats/raw samples without loading OpenVLA.")
    parser.add_argument("--dry-run-samples", type=int, default=4)
    return parser.parse_args()


def init_distributed() -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return device, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def get_attr_from_wrapped(model: nn.Module, name: str) -> Any:
    if hasattr(model, name):
        return getattr(model, name)
    if hasattr(model, "module") and hasattr(model.module, name):
        return getattr(model.module, name)
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, name):
        return getattr(model.base_model.model, name)
    raise AttributeError(f"Could not find attribute {name!r} on {type(model)}")


def count_trainable_parameters(module: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in module.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


class OpenVLAOFTRLBenchTrainingModel(nn.Module):
    def __init__(
        self,
        *,
        vla: nn.Module,
        action_head: nn.Module,
        proprio_projector: nn.Module | None,
        num_patches: int,
        num_actions_chunk: int,
        action_dim: int,
        use_proprio: bool,
        use_wan_steering: bool,
        get_current_action_mask,
        get_next_actions_mask,
    ) -> None:
        super().__init__()
        self.vla = vla
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.num_patches = int(num_patches)
        self.num_actions_chunk = int(num_actions_chunk)
        self.action_dim = int(action_dim)
        self.use_proprio = bool(use_proprio)
        self.use_wan_steering = bool(use_wan_steering)
        self.get_current_action_mask = get_current_action_mask
        self.get_next_actions_mask = get_next_actions_mask

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        device = next(self.parameters()).device
        pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
        input_ids = batch["input_ids"].to(device=device)
        attention_mask = batch["attention_mask"].to(device=device)
        labels = batch["labels"].to(device=device)
        actions_gt = batch["actions"].to(device=device, dtype=torch.bfloat16)
        proprio = None
        if self.use_proprio and self.proprio_projector is not None:
            proprio = batch["proprio"].to(device=device, dtype=torch.bfloat16)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = self.vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=True,
                proprio=proprio,
                proprio_projector=self.proprio_projector if self.use_proprio else None,
                use_film=False,
            )

            ground_truth_token_ids = labels[:, 1:]
            current_action_mask = self.get_current_action_mask(ground_truth_token_ids)
            next_actions_mask = self.get_next_actions_mask(ground_truth_token_ids)
            action_mask = current_action_mask | next_actions_mask

            last_hidden_states = output.hidden_states[-1]
            text_hidden_states = last_hidden_states[:, self.num_patches : -1]
            selected = text_hidden_states[action_mask]
            batch_size = input_ids.shape[0]
            expected = batch_size * self.num_actions_chunk * self.action_dim
            if selected.shape[0] != expected:
                raise RuntimeError(
                    f"OpenVLA action-token mask selected {selected.shape[0]} hidden states, expected {expected}. "
                    "Check action chunk tokenization/constants."
                )
            actions_hidden_states = selected.reshape(batch_size, self.num_actions_chunk * self.action_dim, -1).to(torch.bfloat16)

            if self.use_wan_steering:
                wan_latents = batch["wan_latents"].to(device=device, dtype=torch.bfloat16)
                predicted_actions = self.action_head.predict_action(actions_hidden_states, wan_latents)
            else:
                predicted_actions = self.action_head.predict_action(actions_hidden_states)
            loss = torch.nn.functional.l1_loss(predicted_actions, actions_gt)

        curr_l1 = torch.nn.functional.l1_loss(predicted_actions[:, 0], actions_gt[:, 0]).detach()
        next_l1 = torch.nn.functional.l1_loss(predicted_actions[:, 1:], actions_gt[:, 1:]).detach()
        metrics = {
            "loss": float(loss.detach().float().cpu()),
            "curr_l1": float(curr_l1.float().cpu()),
            "next_l1": float(next_l1.float().cpu()),
        }
        return loss, metrics


def load_openvla_modules(args: argparse.Namespace, device: torch.device, resume_from: Path | None):
    ensure_openvla_oft_on_path(args.openvla_oft_dir)

    from experiments.robot.openvla_utils import check_model_logic_mismatch, model_is_on_hf_hub, update_auto_map
    from peft import LoraConfig, PeftModel, get_peft_model
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    if ACTION_DIM != args.action_dim or NUM_ACTIONS_CHUNK != args.num_actions_chunk:
        raise ValueError(
            f"OpenVLA-OFT constants are ACTION_DIM={ACTION_DIM}, NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}; "
            f"requested action_dim={args.action_dim}, num_actions_chunk={args.num_actions_chunk}."
        )

    if not model_is_on_hf_hub(args.vla_path):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        update_auto_map(args.vla_path)
        check_model_logic_mismatch(args.vla_path)

    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla.vision_backbone.set_num_images_in_input(args.num_images_in_input)
    vla = vla.to(device)

    adapter_dir = resume_from / "lora_adapter" if resume_from is not None else None
    if args.use_lora:
        if adapter_dir is not None and adapter_dir.exists():
            vla = PeftModel.from_pretrained(vla, adapter_dir, is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=min(args.lora_rank, 16),
                lora_dropout=args.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)

    llm_dim = int(get_attr_from_wrapped(vla, "llm_dim"))
    action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=args.action_dim).to(device=device, dtype=torch.bfloat16)
    proprio_projector = None
    if args.use_proprio:
        if args.proprio_dim != 7:
            raise ValueError(f"RLBench proprio_dim should be 7 for absolute rotvec7 state, got {args.proprio_dim}")
        proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=args.proprio_dim).to(device=device, dtype=torch.bfloat16)

    return processor, vla, action_head, proprio_projector


def maybe_load_component_checkpoints(model: OpenVLAOFTRLBenchTrainingModel, resume_from: Path | None, map_location: torch.device) -> int:
    if resume_from is None:
        return 0
    if not resume_from.exists():
        raise FileNotFoundError(f"--resume-from does not exist: {resume_from}")
    if (resume_from / "action_head.pt").exists():
        if isinstance(model.action_head, OpenVLAOFTWanActionHead):
            model.action_head.action_head.load_state_dict(torch.load(resume_from / "action_head.pt", map_location=map_location, weights_only=False))
        else:
            model.action_head.load_state_dict(torch.load(resume_from / "action_head.pt", map_location=map_location, weights_only=False))
    if isinstance(model.action_head, OpenVLAOFTWanActionHead) and (resume_from / "wan_fuser.pt").exists():
        model.action_head.wan_fuser.load_state_dict(torch.load(resume_from / "wan_fuser.pt", map_location=map_location, weights_only=False))
    if model.proprio_projector is not None and (resume_from / "proprio_projector.pt").exists():
        model.proprio_projector.load_state_dict(torch.load(resume_from / "proprio_projector.pt", map_location=map_location, weights_only=False))
    state_path = resume_from / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=map_location, weights_only=False)
        return int(state.get("step", 0))
    return 0


def save_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    processor: Any,
    args: argparse.Namespace,
    stats_path: Path,
    run_dir: Path,
    step: int,
    rank: int,
    world_size: int,
) -> None:
    if world_size > 1:
        dist.barrier()
    if is_main(rank):
        policy = unwrap(model)
        ckpt_dir = run_dir / f"step_{step:08d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(ckpt_dir)
        if args.use_lora:
            policy.vla.save_pretrained(ckpt_dir / "lora_adapter")
        else:
            policy.vla.save_pretrained(ckpt_dir / "vla")

        if isinstance(policy.action_head, OpenVLAOFTWanActionHead):
            torch.save(policy.action_head.action_head.state_dict(), ckpt_dir / "action_head.pt")
            torch.save(policy.action_head.wan_fuser.state_dict(), ckpt_dir / "wan_fuser.pt")
        else:
            torch.save(policy.action_head.state_dict(), ckpt_dir / "action_head.pt")
        if policy.proprio_projector is not None:
            torch.save(policy.proprio_projector.state_dict(), ckpt_dir / "proprio_projector.pt")
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, ckpt_dir / "training_state.pt")
        if stats_path.exists():
            shutil.copy2(stats_path, ckpt_dir / "dataset_statistics.json")
        config = vars(args).copy()
        config["stats_key"] = DATASET_STATS_KEY
        config["step"] = step
        for key, value in list(config.items()):
            if isinstance(value, Path):
                config[key] = value.as_posix()
        (ckpt_dir / "rlbench_openvla_oft_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        logging.info("Saved checkpoint: %s", ckpt_dir)
    if world_size > 1:
        dist.barrier()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    torch.manual_seed(args.seed)
    sample_index = build_or_load_sample_index(
        sample_index_path=args.sample_index_path,
        manifest_path=args.manifest_path,
        event_manifest_path=args.event_manifest_path,
        goal_mode=args.goal_mode,
        split=args.split,
        sample_every_n=args.sample_every_n,
        rgb_root_200=args.rgb_root_200,
        rgb_root_400=args.rgb_root_400,
        rebuild=args.rebuild_sample_index,
    )
    stats = fit_or_load_stats(
        stats_path=args.stats_path,
        sample_index=sample_index,
        lowdim_root_200=args.lowdim_root_200,
        lowdim_root_400=args.lowdim_root_400,
        force_refit=args.refit_stats,
    )

    if args.dry_run:
        dataset = OpenVLAOFTRLBenchDataset(
            sample_index,
            lowdim_root_200=args.lowdim_root_200,
            lowdim_root_400=args.lowdim_root_400,
            stats=stats,
            num_images_in_input=args.num_images_in_input,
            num_actions_chunk=args.num_actions_chunk,
            action_dim=args.action_dim,
            use_proprio=args.use_proprio,
        )
        print(f"sample_index={args.sample_index_path} rows={len(sample_index)}")
        print(f"stats_path={args.stats_path} stats_key={DATASET_STATS_KEY}")
        for idx in range(min(args.dry_run_samples, len(dataset))):
            example = dataset.raw_example(idx)
            record = example["record"]
            print(
                f"[{idx}] task={record.get('task')} src={record.get('source_bundle')} "
                f"ep={record.get('episode')} cur={record.get('frame_index')} "
                f"target={record.get('target_waypoint_frame')} action_norm={example['actions'][0].round(3).tolist()}"
            )
        return

    device, rank, _local_rank, world_size = init_distributed()
    try:
        if is_main(rank):
            logging.info("OpenVLA-OFT RLBench training on device=%s world_size=%s", device, world_size)
            logging.info("Rows: %s", len(sample_index))
        if args.openvla_oft_dir.as_posix():
            ensure_openvla_oft_on_path(args.openvla_oft_dir)

        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction
        from prismatic.vla.action_tokenizer import ActionTokenizer

        processor, vla, base_action_head, proprio_projector = load_openvla_modules(args, device, args.resume_from)
        llm_dim = int(get_attr_from_wrapped(vla, "llm_dim"))
        vision_backbone = get_attr_from_wrapped(vla, "vision_backbone")
        num_patches = int(vision_backbone.get_num_patches() * vision_backbone.get_num_images_in_input())
        if args.use_proprio:
            num_patches += 1

        action_head: nn.Module
        if args.use_wan_steering:
            action_head = OpenVLAOFTWanActionHead(
                base_action_head,
                hidden_dim=llm_dim,
                wan_num_heads=args.wan_fuser_num_heads,
                wan_dropout=args.wan_fuser_dropout,
                use_residual_gate=True,
            ).to(device=device, dtype=torch.bfloat16)
        else:
            action_head = base_action_head

        policy = OpenVLAOFTRLBenchTrainingModel(
            vla=vla,
            action_head=action_head,
            proprio_projector=proprio_projector,
            num_patches=num_patches,
            num_actions_chunk=args.num_actions_chunk,
            action_dim=args.action_dim,
            use_proprio=args.use_proprio,
            use_wan_steering=args.use_wan_steering,
            get_current_action_mask=get_current_action_mask,
            get_next_actions_mask=get_next_actions_mask,
        ).to(device)
        start_step = maybe_load_component_checkpoints(policy, args.resume_from, device)

        expected_wan_shape = parse_latent_shape(args.expected_wan_latent_shape) if args.expected_wan_latent_shape else None
        dataset = OpenVLAOFTRLBenchDataset(
            sample_index,
            lowdim_root_200=args.lowdim_root_200,
            lowdim_root_400=args.lowdim_root_400,
            stats=stats,
            action_tokenizer=ActionTokenizer(processor.tokenizer),
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_cls=PurePromptBuilder,
            num_images_in_input=args.num_images_in_input,
            num_actions_chunk=args.num_actions_chunk,
            action_dim=args.action_dim,
            use_proprio=args.use_proprio,
            use_wan_latents=args.use_wan_steering,
            wan_latent_cache_root=args.wan_latent_cache_root,
            expected_wan_num_inference_steps=args.expected_wan_num_inference_steps,
            expected_wan_backend=args.expected_wan_backend,
            expected_wan_shape=expected_wan_shape,
            allow_missing_wan_latents=args.allow_missing_wan_latents,
        )
        collator = WanLatentCollator(
            PaddedCollatorForActionPrediction(
                model_max_length=processor.tokenizer.model_max_length,
                pad_token_id=processor.tokenizer.pad_token_id,
                padding_side="right",
                pixel_values_dtype=torch.float32,
            )
        )
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if world_size > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collator,
            drop_last=True,
        )

        if world_size > 1:
            policy = DDP(policy, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

        trainable, total = count_trainable_parameters(unwrap(policy))
        if is_main(rank):
            logging.info("Trainable parameters: %.2fM / %.2fM", trainable / 1e6, total / 1e6)
        optimizer = torch.optim.AdamW(
            [p for p in unwrap(policy).parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        if args.resume_from is not None and (args.resume_from / "training_state.pt").exists():
            state = torch.load(args.resume_from / "training_state.pt", map_location=device, weights_only=False)
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])

        run_dir = args.run_root / args.exp_name
        if is_main(rank):
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.stats_path, run_dir / "dataset_statistics.json")

        step = int(start_step)
        optimizer.zero_grad(set_to_none=True)
        while step < args.max_steps:
            if sampler is not None:
                sampler.set_epoch(step)
            for batch_idx, batch in enumerate(loader):
                policy.train()
                loss, metrics = policy(batch)
                loss = loss / args.grad_accumulation_steps
                loss.backward()
                if (batch_idx + 1) % args.grad_accumulation_steps != 0:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(unwrap(policy).parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if is_main(rank) and (step % args.log_interval == 0 or step == 1):
                    logging.info(
                        "step=%s/%s loss=%.5f curr_l1=%.5f next_l1=%.5f grad_norm=%.3f",
                        step,
                        args.max_steps,
                        metrics["loss"],
                        metrics["curr_l1"],
                        metrics["next_l1"],
                        float(grad_norm),
                    )
                if step % args.save_interval == 0 or step >= args.max_steps:
                    save_checkpoint(
                        model=policy,
                        optimizer=optimizer,
                        processor=processor,
                        args=args,
                        stats_path=args.stats_path,
                        run_dir=run_dir,
                        step=step,
                        rank=rank,
                        world_size=world_size,
                    )
                if step >= args.max_steps:
                    break
        if is_main(rank):
            logging.info("Training complete: %s", run_dir)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
