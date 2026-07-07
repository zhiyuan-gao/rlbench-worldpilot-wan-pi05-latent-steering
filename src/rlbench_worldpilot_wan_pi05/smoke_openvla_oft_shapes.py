from __future__ import annotations

import argparse

import torch
from torch import nn

from .openvla_oft_steering import OpenVLAOFTWanActionHead


class DummyL1ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, chunk_len: int, action_dim: int) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.proj = nn.Linear(hidden_dim * action_dim, action_dim)

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        bsz = actions_hidden_states.shape[0]
        grouped = actions_hidden_states.reshape(bsz, self.chunk_len, self.action_dim * actions_hidden_states.shape[-1])
        return self.proj(grouped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the OpenVLA-OFT WAN latent steering action-head wrapper.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-len", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--latent-steps", type=int, default=6)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    action_head = DummyL1ActionHead(args.hidden_dim, args.chunk_len, args.action_dim).to(dtype=dtype)
    model = OpenVLAOFTWanActionHead(
        action_head,
        hidden_dim=args.hidden_dim,
        wan_num_heads=args.num_heads,
        use_residual_gate=True,
    ).to(dtype=dtype)

    actions_hidden = torch.randn(
        args.batch_size,
        args.chunk_len * args.action_dim,
        args.hidden_dim,
        dtype=dtype,
    )
    wan_latents = torch.randn(
        args.batch_size,
        args.views,
        args.channels,
        args.latent_steps,
        args.height,
        args.width,
        dtype=dtype,
    )
    with torch.no_grad():
        fused = model.fuse_action_hidden_states(actions_hidden, wan_latents)
        actions = model.predict_action(actions_hidden, wan_latents)

    max_abs_delta = (fused.float() - actions_hidden.float()).abs().max().item()
    print(f"actions_hidden_states: {tuple(actions_hidden.shape)} dtype={actions_hidden.dtype}")
    print(f"wan_latents:           {tuple(wan_latents.shape)} dtype={wan_latents.dtype}")
    print(f"fused_hidden_states:   {tuple(fused.shape)}")
    print(f"pred_actions:          {tuple(actions.shape)}")
    print(f"initial_gate_delta:    {max_abs_delta:.6g}")


if __name__ == "__main__":
    main()
