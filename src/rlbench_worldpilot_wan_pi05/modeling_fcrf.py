from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from .fcrf import FCRFResidualFlow


class PI0FCRFV1Pytorch(PI0Pytorch):
    """Frozen RLBench pi0.5-ft plus a WAN-conditioned action-flow residual."""

    def __init__(
        self,
        config,
        *,
        fcrf_num_heads: int = 8,
        fcrf_dropout: float = 0.0,
        fcrf_gate_bias: float = -2.2,
    ) -> None:
        super().__init__(config)
        action_hidden_dim = int(self.action_in_proj.out_features)
        self.fcrf = FCRFResidualFlow(
            action_hidden_dim=action_hidden_dim,
            action_dim=int(config.action_dim),
            num_heads=int(fcrf_num_heads),
            dropout=float(fcrf_dropout),
            gate_bias=float(fcrf_gate_bias),
        )

    def initialize_fcrf(
        self,
        wan_latents: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Materialize the lazy WAN projection before loading/optimizing."""
        self.fcrf.to(device=device, dtype=dtype)
        dummy_hidden = torch.zeros(
            wan_latents.shape[0],
            int(self.config.action_horizon),
            int(self.action_in_proj.out_features),
            device=device,
            dtype=dtype,
        )
        dummy_time = torch.ones(wan_latents.shape[0], device=device, dtype=torch.float32)
        self.fcrf(dummy_hidden, wan_latents.to(device=device, dtype=dtype), dummy_time)

    def freeze_base(self) -> tuple[list[str], list[str]]:
        """Freeze every inherited pi0.5 parameter and train only ``fcrf.*``."""
        trainable = []
        frozen = []
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("fcrf."))
            (trainable if parameter.requires_grad else frozen).append(name)
        return trainable, frozen

    def train(self, mode: bool = True):
        super().train(mode)
        # The base is a deterministic frozen teacher even while FCRF trains.
        self.paligemma_with_expert.eval()
        self.action_in_proj.eval()
        self.action_out_proj.eval()
        if self.pi05:
            self.time_mlp_in.eval()
            self.time_mlp_out.eval()
        self.fcrf.train(mode)
        return self

    def compute_base_outputs(
        self,
        observation,
        actions: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        preprocess_train: bool = True,
    ) -> dict[str, Tensor]:
        """Run the frozen base once and expose its suffix hidden and flow."""
        with torch.no_grad():
            images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
                observation,
                train=preprocess_train,
            )
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)
            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            target_flow = noise - actions

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
            )
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state,
                x_t,
                time,
            )
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
            base_flow = self.action_out_proj(suffix_out)

        return {
            "suffix_out_base": suffix_out.detach(),
            "base_flow": base_flow.detach(),
            "target_flow": target_flow.detach(),
            "noise": noise.detach(),
            "time": time.detach(),
        }

    def apply_fcrf(
        self,
        base_outputs: dict[str, Tensor],
        wan_latents: Tensor | None,
        *,
        enabled: bool = True,
    ) -> dict[str, Tensor]:
        base_flow = base_outputs["base_flow"]
        target_flow = base_outputs["target_flow"]
        if not enabled or wan_latents is None:
            correction = torch.zeros_like(base_flow)
            delta_flow = torch.zeros_like(base_flow)
            gate = torch.zeros(
                base_flow.shape[0],
                1,
                1,
                device=base_flow.device,
                dtype=torch.float32,
            )
        else:
            if wan_latents.ndim == 5:
                wan_latents = wan_latents.unsqueeze(0)
            fcrf_out = self.fcrf(
                base_outputs["suffix_out_base"],
                wan_latents.to(device=base_flow.device),
                base_outputs["time"],
                latent_layout="bvcthw",
            )
            correction = fcrf_out.correction
            delta_flow = fcrf_out.delta_flow
            gate = fcrf_out.gate

        final_flow = base_flow + correction
        flow_loss = F.mse_loss(target_flow, final_flow, reduction="none")
        residual_penalty = correction.square().mean(dim=(1, 2))
        desired_correction = target_flow - base_flow
        correction_cosine = F.cosine_similarity(
            correction.flatten(1),
            desired_correction.flatten(1),
            dim=1,
            eps=1e-8,
        )
        return {
            **base_outputs,
            "delta_flow": delta_flow,
            "gate": gate,
            "correction": correction,
            "final_flow": final_flow,
            "flow_loss": flow_loss,
            "residual_penalty": residual_penalty,
            "correction_cosine": correction_cosine,
        }

    def forward(
        self,
        observation,
        actions,
        wan_latents=None,
        noise=None,
        time=None,
        *,
        fcrf_enabled: bool = True,
    ) -> dict[str, Tensor]:
        base_outputs = self.compute_base_outputs(
            observation,
            actions,
            noise=noise,
            time=time,
            preprocess_train=self.training,
        )
        return self.apply_fcrf(base_outputs, wan_latents, enabled=fcrf_enabled)

    def denoise_step_with_hidden(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state,
                x_t,
                timestep,
            )
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
            base_flow = self.action_out_proj(suffix_out)
        return suffix_out.detach(), base_flow.detach()

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        wan_latents=None,
        noise=None,
        num_steps=10,
        *,
        fcrf_enabled: bool = True,
    ) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation,
            train=False,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            suffix_out, base_flow = self.denoise_step_with_hidden(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            if fcrf_enabled and wan_latents is not None:
                fcrf_out = self.fcrf(suffix_out, wan_latents, expanded_time)
                flow = base_flow + fcrf_out.correction
            else:
                flow = base_flow
            x_t = x_t + dt * flow
            time += dt
        return x_t
