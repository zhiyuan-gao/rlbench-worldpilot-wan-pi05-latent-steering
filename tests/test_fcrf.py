from __future__ import annotations

from collections import defaultdict
import math

from rlbench_worldpilot_wan_pi05.fcrf import FCRFResidualFlow
from rlbench_worldpilot_wan_pi05.fcrf import WanFutureTokenEncoder
from rlbench_worldpilot_wan_pi05.train_fcrf_v1 import SameSkillShuffler
from rlbench_worldpilot_wan_pi05.train_fcrf_v1 import _finalize_metrics
from rlbench_worldpilot_wan_pi05.train_fcrf_v1 import _update_metrics
import torch


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


def test_same_skill_shuffle_is_event_matched_and_different_episode() -> None:
    class Dataset:
        def __init__(self) -> None:
            self.sample_index = [
                {
                    "lerobot_index": episode * 2 + event,
                    "task": "task_a",
                    "event_idx": event,
                    "source_bundle": "all200",
                    "variation": "variation0",
                    "episode": f"episode{episode}",
                }
                for episode in range(4)
                for event in range(2)
            ]

    dataset = Dataset()
    shuffler = SameSkillShuffler(dataset, seed=0)
    assert shuffler.task_fallback == 0
    assert shuffler.event_matched == len(dataset.sample_index)
    for row in dataset.sample_index:
        target = shuffler.mapping[row["lerobot_index"]]
        assert target["task"] == row["task"]
        assert target["event_idx"] == row["event_idx"]
        assert target["episode"] != row["episode"]


def test_flow_diagnostics_separate_physical_actions_from_padding() -> None:
    class Shuffler:
        @staticmethod
        def task_for(_index: int) -> str:
            return "task_a"

    target_flow = torch.zeros(1, 1, 32)
    target_flow[..., :7] = 1.0
    base_flow = torch.zeros_like(target_flow)
    correction = torch.zeros_like(target_flow)
    correction[..., :7] = 1.0
    matched_loss = torch.zeros_like(target_flow)
    shuffled_loss = torch.zeros_like(target_flow)
    shuffled_loss[..., :7] = 1.0
    raw = {
        "overall": defaultdict(float),
        "by_task": {"task_a": defaultdict(float)},
    }
    _update_metrics(
        raw,
        {"target_flow": target_flow, "base_flow": base_flow},
        {
            "flow_loss": matched_loss,
            "correction": correction,
            "delta_flow": correction,
            "gate": torch.full((1, 1, 1), 0.25),
        },
        {"flow_loss": shuffled_loss},
        torch.tensor([9]),
        Shuffler(),
    )
    summary = _finalize_metrics(raw)["overall"]

    assert summary["off_mse"] == 7 / 32
    assert summary["physical_off_mse"] == 1.0
    assert summary["physical_matched_mse"] == 0.0
    assert summary["physical_shuffled_mse"] == 1.0
    assert math.isclose(summary["physical_correction_cosine"], 1.0, rel_tol=1e-6)
    assert summary["physical_matched_improvement_over_off"] == 1.0
