from __future__ import annotations

import json
import os

from rlbench_worldpilot_wan_pi05.fcrf import FCRFResidualFlow
import torch
import torch.distributed as dist


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("FCRF DDP smoke requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method="env://", device_id=device)
    torch.manual_seed(1234 + local_rank)

    module = FCRFResidualFlow(action_hidden_dim=32, action_dim=32, num_heads=4).to(
        device=device,
        dtype=torch.bfloat16,
    )
    # Materialize LazyLinear before DDP, matching the real trainer.
    with torch.no_grad():
        module(
            torch.zeros(2, 1, 32, device=device, dtype=torch.bfloat16),
            torch.zeros(2, 3, 2, 6, 2, 2, device=device, dtype=torch.bfloat16),
            torch.ones(2, device=device),
        )
    ddp = torch.nn.parallel.DistributedDataParallel(
        module,
        device_ids=[local_rank],
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
    suffix = torch.randn(2, 1, 32, device=device, dtype=torch.bfloat16)
    latents = torch.randn(2, 3, 2, 6, 2, 2, device=device, dtype=torch.bfloat16)
    flow_time = torch.rand(2, device=device)
    target = torch.randn(2, 1, 32, device=device)
    output = ddp(suffix, latents, flow_time)
    loss = (output.correction - target).square().mean()
    loss.backward()
    optimizer.step()

    checksum = module.residual_out.weight.float().sum()
    gathered = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, checksum)
    if not all(torch.equal(gathered[0], value) for value in gathered[1:]):
        raise RuntimeError(f"DDP parameters diverged after one step: {gathered}")
    if local_rank == 0:
        print(
            json.dumps(
                {
                    "smoke": "passed",
                    "world_size": dist.get_world_size(),
                    "dtype": "bfloat16",
                    "residual_out_checksum": float(checksum.cpu()),
                },
                sort_keys=True,
            )
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
