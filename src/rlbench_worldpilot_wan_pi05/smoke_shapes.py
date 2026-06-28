from __future__ import annotations

import argparse

import torch

from rlbench_worldpilot_wan_pi05.fusion import WanFutureVideoFuser


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test WAN latent fuser tensor shapes.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--latent-steps", type=int, default=6)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--layout", choices=["bvcthw", "bvtchw", "bcthw", "btchw"], default="bvcthw")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    vlm_hidden = torch.randn(args.batch_size, args.seq_len, args.hidden_dim, device=device)

    if args.layout == "bvcthw":
        latents = torch.randn(
            args.batch_size,
            args.views,
            args.channels,
            args.latent_steps,
            args.height,
            args.width,
            device=device,
        )
    elif args.layout == "bvtchw":
        latents = torch.randn(
            args.batch_size,
            args.views,
            args.latent_steps,
            args.channels,
            args.height,
            args.width,
            device=device,
        )
    elif args.layout == "bcthw":
        latents = torch.randn(
            args.batch_size,
            args.channels,
            args.latent_steps,
            args.height,
            args.width,
            device=device,
        )
    else:
        latents = torch.randn(
            args.batch_size,
            args.latent_steps,
            args.channels,
            args.height,
            args.width,
            device=device,
        )

    fuser = WanFutureVideoFuser(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
    ).to(device)
    output = fuser(vlm_hidden, latents, latent_layout=args.layout)

    print(f"vlm_hidden:    {tuple(vlm_hidden.shape)}")
    print(f"wan_latents:   {tuple(latents.shape)} layout={args.layout}")
    print(f"latent_tokens: {tuple(output.latent_tokens.shape)}")
    print(f"fused_hidden:  {tuple(output.hidden_states.shape)}")


if __name__ == "__main__":
    main()
