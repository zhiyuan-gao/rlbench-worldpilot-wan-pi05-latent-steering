from __future__ import annotations

import math

import torch

from rlbench_worldpilot_wan_pi05.fcrf import FCRFResidualFlow, WanFutureTokenEncoder


def test_wan_token_encoder_preserves_view_and_time_axes() -> None:
    encoder = WanFutureTokenEncoder(hidden_dim=16, max_views=4, max_latent_steps=8)
    latents = torch.randn(2, 3, 2, 6, 2, 2)
    tokens = encoder(latents)
    assert tokens.shape == (2, 18, 16)


def test_fcrf_is_exactly_base_equivalent_at_initialization() -> None:
    module = FCRFResidualFlow(
        action_hidden_dim=16,
        action_dim=7,
        num_heads=4,
        gate_bias=-2.2,
    )
    suffix = torch.randn(2, 1, 16)
    latents = torch.randn(2, 3, 2, 6, 2, 2)
    flow_time = torch.tensor([0.25, 0.75])
    output = module(suffix, latents, flow_time)

    assert output.latent_tokens.shape == (2, 18, 16)
    assert torch.count_nonzero(output.delta_flow) == 0
    assert torch.count_nonzero(output.correction) == 0
    assert torch.allclose(
        output.gate,
        torch.full_like(output.gate, 1.0 / (1.0 + math.exp(2.2))),
    )


def test_initial_backward_updates_only_the_residual_output_layer() -> None:
    module = FCRFResidualFlow(action_hidden_dim=16, action_dim=7, num_heads=4)
    suffix = torch.randn(2, 1, 16)
    latents = torch.randn(2, 3, 2, 6, 2, 2)
    flow_time = torch.tensor([0.25, 0.75])
    target = torch.randn(2, 1, 7)
    output = module(suffix, latents, flow_time)
    loss = (output.correction - target).square().mean()
    loss.backward()

    assert module.residual_out.weight.grad is not None
    assert torch.count_nonzero(module.residual_out.weight.grad) > 0
    # A zero-initialized residual deliberately gives the gate no first-step
    # gradient; it starts learning once residual_out becomes non-zero.
    assert module.gate_mlp[-1].weight.grad is not None
    assert torch.count_nonzero(module.gate_mlp[-1].weight.grad) == 0
