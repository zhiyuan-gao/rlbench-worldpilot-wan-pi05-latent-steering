from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch training entry for WAN pi0.5 latent steering.")
    parser.add_argument("--exp-name", default=os.environ.get("EXP_NAME", "selected10_worldpilot_wan_pi05_torch"))
    parser.add_argument("--lerobot-repo-id", default=os.environ.get("LEROBOT_REPO_ID", "rlbench/selected10_pi05_waypoint_h1"))
    parser.add_argument("--manifest-path", default=os.environ.get("MANIFEST_PATH"))
    parser.add_argument("--wan-latent-cache-root", default=os.environ.get("WAN_LATENT_CACHE_ROOT"))
    parser.add_argument("--pytorch-weight-path", default=os.environ.get("PYTORCH_WEIGHT_PATH"))
    parser.add_argument("--time-mode", choices=["all", "last", "mean"], default=os.environ.get("WAN_LATENT_TIME_MODE", "all"))
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()

    print("WAN pi0.5 latent steering PyTorch entry")
    print(f"  exp_name:              {args.exp_name}")
    print(f"  lerobot_repo_id:       {args.lerobot_repo_id}")
    print(f"  manifest_path:         {args.manifest_path}")
    print(f"  wan_latent_cache_root: {args.wan_latent_cache_root}")
    print(f"  pytorch_weight_path:   {args.pytorch_weight_path}")
    print(f"  time_mode:             {args.time_mode}")
    print(f"  extra_args:            {extra_args}")

    if args.dry_run:
        print("Dry run complete. Training loop is intentionally not launched.")
        return

    raise SystemExit(
        "Training loop is not implemented yet. Next step: patch OpenPI PyTorch pi0.5 to call "
        "WanFutureVideoFuser at the VLM hidden-state injection point."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

