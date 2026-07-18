from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import pickle
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from .online_wan import WanDiffusersOnlineProvider
from .rpc_numpy import decode_arrays, encode_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP RPC server for FLF WAN latent generation.")
    parser.add_argument("--host", default=os.environ.get("WAN_RPC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WAN_RPC_PORT", "18766")))
    parser.add_argument("--wan-base-model", type=Path, default=os.environ.get("WAN_BASE_MODEL"), required=os.environ.get("WAN_BASE_MODEL") is None)
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
    print("Loading WAN FLF latent provider...", flush=True)
    provider = WanDiffusersOnlineProvider(
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
    print("Loaded WAN FLF latent provider", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *values: Any) -> None:
            print(f"{self.address_string()} - {fmt % values}", flush=True)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                write_response(self, 200, {"ok": True})
                return
            write_response(self, 404, {"ok": False, "error": f"unknown path {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/latent":
                write_response(self, 404, {"ok": False, "error": f"unknown path {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = decode_arrays(pickle.loads(self.rfile.read(length)))
                latents = provider(
                    request["current_images"],
                    request["goal_images"],
                    prompt=str(request.get("prompt") or ""),
                    seed=int(request.get("seed", 0)),
                )
                latents_np = np.asarray(latents.detach().cpu(), dtype=np.float16)
                write_response(self, 200, encode_arrays({"ok": True, "latents": latents_np}))
            except Exception as exc:
                print(f"WAN latent RPC request failed: {exc!r}\n{traceback.format_exc(limit=10)}", flush=True)
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
    print(f"Serving WAN latent RPC on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
