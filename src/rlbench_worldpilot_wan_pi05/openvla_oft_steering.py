from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn

from .fusion import WanFutureVideoFuser


@dataclass(frozen=True)
class OpenVLAOFTSteeringConfig:
    """Configuration for the OpenVLA-OFT WAN latent steering path."""

    checkpoint: str
    openvla_oft_dir: str | Path
    num_actions_chunk: int = 8
    action_dim: int = 7
    proprio_dim: int = 7
    num_images_in_input: int = 3
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    use_proprio: bool = True
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    unnorm_key: str = ""


def ensure_openvla_oft_on_path(openvla_oft_dir: str | Path) -> Path:
    """Add an external OpenVLA-OFT checkout to ``sys.path``.

    This repo intentionally does not vendor OpenVLA-OFT.  Training/inference code
    should point ``OPENVLA_OFT_DIR`` to an external checkout on the HPC.
    """

    root = Path(openvla_oft_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"OPENVLA_OFT_DIR does not exist: {root}. Clone https://github.com/moojink/openvla-oft first."
        )
    if not (root / "experiments" / "robot" / "openvla_utils.py").exists():
        raise FileNotFoundError(f"OpenVLA-OFT checkout looks incomplete: {root}")
    root_str = root.as_posix()
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def make_openvla_oft_generate_config(config: OpenVLAOFTSteeringConfig) -> Any:
    """Create OpenVLA-OFT's ``GenerateConfig`` lazily.

    Importing OpenVLA-OFT is optional for local shape smoke tests.  Real OpenVLA
    loading happens only when this function is called.
    """

    ensure_openvla_oft_on_path(config.openvla_oft_dir)
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    return GenerateConfig(
        pretrained_checkpoint=config.checkpoint,
        use_l1_regression=config.use_l1_regression,
        use_diffusion=config.use_diffusion,
        use_film=config.use_film,
        num_images_in_input=config.num_images_in_input,
        use_proprio=config.use_proprio,
        load_in_8bit=config.load_in_8bit,
        load_in_4bit=config.load_in_4bit,
        center_crop=config.center_crop,
        num_open_loop_steps=config.num_actions_chunk,
        unnorm_key=config.unnorm_key,
    )


class OpenVLAOFTWanActionHead(nn.Module):
    """WorldPilot-style WAN residual before the OpenVLA-OFT continuous action head.

    OpenVLA-OFT exposes final hidden states for action tokens:

    ``actions_hidden_states: (B, chunk_len * action_dim, D)``

    Its L1 action head maps those hidden states to a continuous action chunk.  We
    insert the same WAN future latent fuser used by the pi0.5 path between those
    action-token hidden states and the OpenVLA-OFT action head.
    """

    def __init__(
        self,
        action_head: nn.Module,
        *,
        hidden_dim: int,
        wan_num_heads: int = 8,
        wan_dropout: float = 0.0,
        use_residual_gate: bool = True,
    ) -> None:
        super().__init__()
        self.action_head = action_head
        self.wan_fuser = WanFutureVideoFuser(
            hidden_dim=hidden_dim,
            num_heads=wan_num_heads,
            dropout=wan_dropout,
            use_residual_gate=use_residual_gate,
            use_post_norm=not use_residual_gate,
        )

    def initialize_wan_fuser(self, wan_latents: torch.Tensor, *, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        self.wan_fuser.to(device=device, dtype=dtype)
        dummy_hidden = torch.zeros(wan_latents.shape[0], 1, hidden_dim, device=device, dtype=dtype)
        self.wan_fuser(dummy_hidden, wan_latents.to(device=device, dtype=dtype), latent_layout="bvcthw")

    def fuse_action_hidden_states(self, actions_hidden_states: torch.Tensor, wan_latents: torch.Tensor | None) -> torch.Tensor:
        if wan_latents is None:
            return actions_hidden_states
        if wan_latents.ndim == 5:
            wan_latents = wan_latents.unsqueeze(0)
        return self.wan_fuser(
            actions_hidden_states,
            wan_latents.to(device=actions_hidden_states.device),
            latent_layout="bvcthw",
        ).hidden_states

    def predict_action(self, actions_hidden_states: torch.Tensor, wan_latents: torch.Tensor | None = None) -> torch.Tensor:
        fused_hidden = self.fuse_action_hidden_states(actions_hidden_states, wan_latents)
        if hasattr(self.action_head, "predict_action"):
            return self.action_head.predict_action(fused_hidden)
        return self.action_head(fused_hidden)
