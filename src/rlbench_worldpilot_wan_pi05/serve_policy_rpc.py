from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import pickle
from pathlib import Path
import traceback
from typing import Any
from urllib.error import HTTPError
from urllib import request as urlrequest

import numpy as np

from .online_policy import WanPi05OnlinePolicy
from .online_wan import DummyWanLatentProvider, WanDiffusersOnlineProvider, parse_latent_shape
from .rpc_numpy import decode_arrays, encode_arrays


def make_wan_provider(args: argparse.Namespace):
    if args.wan_rpc_url:
        return WanRpcLatentProvider(args.wan_rpc_url, timeout_sec=args.wan_rpc_timeout_sec)
    if args.wan_backend == "dummy":
        return DummyWanLatentProvider(parse_latent_shape(args.wan_latent_shape))
    if args.wan_base_model is None:
        raise ValueError("--wan-base-model is required when --wan-backend=wan-diffusers")
    return WanDiffusersOnlineProvider(
        base_model=args.wan_base_model,
        lora_dir=args.wan_lora_dir,
        height=args.wan_height,
        view_width=args.wan_view_width,
        num_views=3,
        num_frames=args.wan_num_frames,
        num_inference_steps=args.wan_num_inference_steps,
        guidance_scale=args.wan_guidance_scale,
        lora_scale=args.wan_lora_scale,
        dtype=args.wan_dtype,
        device_map=args.wan_device_map,
        output_layout=args.wan_output_layout,
    )


class WanRpcLatentProvider:
    def __init__(self, url: str, *, timeout_sec: float = 900.0) -> None:
        self.url = url
        self.timeout_sec = float(timeout_sec)

    def __call__(
        self,
        current_images,
        goal_images,
        *,
        prompt: str,
        seed: int = 0,
    ):
        body = pickle.dumps(
            encode_arrays(
            {
                "current_images": current_images,
                "goal_images": goal_images,
                "prompt": prompt,
                "seed": int(seed),
            }
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        req = urlrequest.Request(self.url, data=body, method="POST")
        req.add_header("Content-Type", "application/x-python-pickle")
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_sec) as response:
                result = decode_arrays(pickle.loads(response.read()))
        except HTTPError as exc:
            try:
                result = decode_arrays(pickle.loads(exc.read()))
            except Exception as read_exc:
                raise RuntimeError(f"WAN RPC HTTP {exc.code}; could not decode error body: {read_exc!r}") from exc
        if not result.get("ok"):
            raise RuntimeError(f"WAN RPC failed: {result.get('error')}\n{result.get('traceback', '')}")
        return np.asarray(result["latents"], dtype=np.float16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP RPC server for WorldPilot WAN pi0.5 policy inference.")
    parser.add_argument("config_name", nargs="?", default=os.environ.get("CONFIG_NAME", "pi05_rlbench_waypoint_h1"))
    parser.add_argument("--host", default=os.environ.get("POLICY_RPC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("POLICY_RPC_PORT", "8765")))
    parser.add_argument("--exp-name", default=os.environ.get("EXP_NAME", "selected10_worldpilot_wan_pi05_torch"))
    parser.add_argument("--eval-checkpoint", default=os.environ.get("EVAL_CHECKPOINT"))
    parser.add_argument("--checkpoint-base-dir", default=os.environ.get("CHECKPOINT_BASE_DIR", "./checkpoints"))
    parser.add_argument("--assets-base-dir", default=os.environ.get("ASSETS_BASE_DIR", "./assets"))

    parser.add_argument("--policy-device", default=os.environ.get("POLICY_DEVICE", "cuda:0"))
    parser.add_argument(
        "--pytorch-training-precision",
        choices=("bfloat16", "float32"),
        default=os.environ.get("PYTORCH_TRAINING_PRECISION"),
    )
    parser.add_argument("--wan-latent-shape", default=os.environ.get("WAN_LATENT_SHAPE", "3,16,6,32,32"))
    parser.add_argument("--wan-fuser-num-heads", type=int, default=int(os.environ.get("WAN_FUSER_NUM_HEADS", "8")))
    parser.add_argument("--wan-fuser-dropout", type=float, default=float(os.environ.get("WAN_FUSER_DROPOUT", "0.0")))
    parser.add_argument("--wan-steering-mode", choices=("early", "block"), default=os.environ.get("WAN_STEERING_MODE", "early"))
    parser.add_argument("--wan-steering-block", type=int, default=int(os.environ.get("WAN_STEERING_BLOCK", "12")))
    parser.add_argument("--wan-steering-gate", choices=("auto", "on", "off"), default=os.environ.get("WAN_STEERING_GATE", "auto"))
    parser.add_argument("--action-num-steps", type=int, default=10)

    parser.add_argument("--wan-backend", choices=("dummy", "wan-diffusers"), default=os.environ.get("WAN_LATENT_BACKEND", "dummy"))
    parser.add_argument("--wan-rpc-url", default=os.environ.get("WAN_RPC_URL"))
    parser.add_argument("--wan-rpc-timeout-sec", type=float, default=float(os.environ.get("WAN_RPC_TIMEOUT_SEC", "900")))
    parser.add_argument("--wan-base-model", type=Path, default=os.environ.get("WAN_BASE_MODEL"))
    parser.add_argument("--wan-lora-dir", type=Path, default=os.environ.get("WAN_LORA_DIR") or None)
    parser.add_argument("--wan-height", type=int, default=256)
    parser.add_argument("--wan-view-width", type=int, default=256)
    parser.add_argument("--wan-num-frames", type=int, default=21)
    parser.add_argument("--wan-output-layout", choices=("bcthw", "btchw"), default=os.environ.get("WAN_OUTPUT_LAYOUT", "bcthw"))
    parser.add_argument("--wan-num-inference-steps", type=int, default=int(os.environ.get("WAN_NUM_INFERENCE_STEPS", "1")))
    parser.add_argument("--wan-guidance-scale", type=float, default=1.0)
    parser.add_argument("--wan-lora-scale", type=float, default=1.0)
    parser.add_argument("--wan-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--wan-device-map", default=os.environ.get("WAN_DEVICE_MAP", "balanced"))
    return parser.parse_args()


def write_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    handler.send_response(status)
    handler.send_header("Content-Type", "application/x-python-pickle")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def main() -> None:
    args = parse_args()
    print("Loading WorldPilot policy...", flush=True)
    policy = WanPi05OnlinePolicy(
        config_name=args.config_name,
        exp_name=args.exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        assets_base_dir=args.assets_base_dir,
        eval_checkpoint=args.eval_checkpoint,
        wan_latent_shape=parse_latent_shape(args.wan_latent_shape),
        device=args.policy_device,
        precision=args.pytorch_training_precision,
        wan_num_heads=args.wan_fuser_num_heads,
        wan_dropout=args.wan_fuser_dropout,
        wan_steering_mode=args.wan_steering_mode,
        wan_steering_block=args.wan_steering_block,
        wan_steering_gate=args.wan_steering_gate,
        action_num_steps=args.action_num_steps,
    )
    print(f"Loaded policy checkpoint: {policy.checkpoint_dir}", flush=True)

    print("Loading WAN provider...", flush=True)
    wan_provider = make_wan_provider(args)
    print(f"Loaded WAN backend: {args.wan_backend}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *values: Any) -> None:
            print(f"{self.address_string()} - {fmt % values}", flush=True)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                write_response(self, 200, {"ok": True, "checkpoint": policy.checkpoint_dir.as_posix()})
                return
            write_response(self, 404, {"ok": False, "error": f"unknown path {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/infer":
                write_response(self, 404, {"ok": False, "error": f"unknown path {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = decode_arrays(pickle.loads(self.rfile.read(length)))
                wan_mode = str(request.get("wan_mode", "matched"))
                if wan_mode not in ("matched", "off"):
                    raise ValueError(f"Unsupported wan_mode={wan_mode!r}")
                wan_seed = int(request.get("wan_seed", request.get("seed", 0)))
                action_seed = request.get("action_seed")
                wan_latents = None
                if wan_mode == "matched":
                    wan_latents = wan_provider(
                        request["current_images"],
                        request["goal_images"],
                        prompt=str(request.get("wan_prompt") or request.get("prompt") or ""),
                        seed=wan_seed,
                    )
                action7 = policy.infer_action7(
                    request["policy_obs"],
                    wan_latents,
                    action_seed=None if action_seed is None else int(action_seed),
                )
                write_response(
                    self,
                    200,
                    encode_arrays(
                        {
                            "ok": True,
                            "action7": np.asarray(action7, dtype=np.float32),
                            "wan_mode": wan_mode,
                            "wan_seed": wan_seed,
                            "action_seed": action_seed,
                        }
                    ),
                )
            except Exception as exc:  # keep server alive across bad rollout requests
                print(f"Policy RPC request failed: {exc!r}\n{traceback.format_exc(limit=10)}", flush=True)
                write_response(
                    self,
                    500,
                    {
                        "ok": False,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=10),
                    },
                )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving policy RPC on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
