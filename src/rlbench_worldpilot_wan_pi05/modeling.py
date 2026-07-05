from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers.models.gemma import modeling_gemma
from typing import Literal

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from .fusion import WanFutureVideoFuser


WanSteeringMode = Literal["early", "block"]
WanSteeringGate = Literal["auto", "on", "off"]


class PI0WanLatentSteeringPytorch(PI0Pytorch):
    """pi0.5 PyTorch model with WorldPilot-style WAN latent steering."""

    def __init__(
        self,
        config,
        *,
        wan_num_heads: int = 8,
        wan_dropout: float = 0.0,
        wan_steering_mode: WanSteeringMode = "early",
        wan_steering_block: int = 12,
        wan_steering_gate: WanSteeringGate = "auto",
    ) -> None:
        super().__init__(config)
        if wan_steering_mode not in ("early", "block"):
            raise ValueError(f"Unsupported wan_steering_mode={wan_steering_mode!r}")
        if wan_steering_gate not in ("auto", "on", "off"):
            raise ValueError(f"Unsupported wan_steering_gate={wan_steering_gate!r}")
        hidden_dim = int(self.paligemma_with_expert.paligemma.config.text_config.hidden_size)
        num_layers = int(self.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers)
        if not 1 <= int(wan_steering_block) <= num_layers:
            raise ValueError(f"wan_steering_block must be in [1, {num_layers}], got {wan_steering_block}")
        self.wan_steering_mode = wan_steering_mode
        self.wan_steering_block = int(wan_steering_block)
        self.wan_steering_gate = wan_steering_gate
        use_residual_gate = wan_steering_gate == "on" or (
            wan_steering_gate == "auto" and wan_steering_mode == "block"
        )
        self.wan_fuser = WanFutureVideoFuser(
            hidden_dim=hidden_dim,
            num_heads=wan_num_heads,
            dropout=wan_dropout,
            use_residual_gate=use_residual_gate,
            use_post_norm=not use_residual_gate,
        )

    def initialize_wan_fuser(self, wan_latents: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> None:
        """Materialize lazy fuser parameters before optimizer construction."""
        hidden_dim = int(self.paligemma_with_expert.paligemma.config.text_config.hidden_size)
        self.wan_fuser.to(device=device, dtype=dtype)
        dummy_hidden = torch.zeros(wan_latents.shape[0], 1, hidden_dim, device=device, dtype=dtype)
        self.wan_fuser(dummy_hidden, wan_latents.to(device=device, dtype=dtype), latent_layout="bvcthw")

    def fuse_hidden_with_wan(self, hidden_states: Tensor, wan_latents: Tensor | None) -> Tensor:
        if wan_latents is None:
            return hidden_states
        if wan_latents.ndim == 5:
            wan_latents = wan_latents.unsqueeze(0)
        return self.wan_fuser(
            hidden_states,
            wan_latents.to(device=hidden_states.device),
            latent_layout="bvcthw",
        ).hidden_states

    def fuse_prefix_with_wan(self, prefix_embs: Tensor, wan_latents: Tensor | None) -> Tensor:
        return self.fuse_hidden_with_wan(prefix_embs, wan_latents)

    def _forward_with_block_steering(
        self,
        prefix_embs: Tensor,
        suffix_embs: Tensor,
        att_2d_masks_4d: Tensor,
        position_ids: Tensor,
        adarms_cond: Tensor | None,
        wan_latents: Tensor | None,
    ):
        """Run pi0.5 joint prefix/suffix layers and inject WAN after a prefix block."""
        models = [
            self.paligemma_with_expert.paligemma.language_model,
            self.paligemma_with_expert.gemma_expert.model,
        ]
        num_layers = int(self.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers)
        conds = [None, adarms_cond]
        use_gradient_checkpointing = (
            self.training
            and hasattr(self.paligemma_with_expert.gemma_expert.model, "gradient_checkpointing")
        )
        if use_gradient_checkpointing:
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, conds):
            query_states = []
            key_states = []
            value_states = []
            gates = []
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]
                hidden_states, gate = layer.input_layernorm(hidden_states, cond=conds[i])
                gates.append(gate)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            query_states = torch.cat(query_states, dim=2)
            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)

            dummy_tensor = torch.zeros(
                query_states.shape[0],
                query_states.shape[2],
                query_states.shape[-1],
                device=query_states.device,
                dtype=query_states.dtype,
            )
            cos, sin = self.paligemma_with_expert.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            batch_size = query_states.shape[0]
            scaling = self.paligemma_with_expert.paligemma.language_model.layers[layer_idx].self_attn.scaling
            att_output, _ = modeling_gemma.eager_attention_forward(
                self.paligemma_with_expert.paligemma.language_model.layers[layer_idx].self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling,
            )
            head_dim = self.paligemma_with_expert.paligemma.language_model.layers[layer_idx].self_attn.head_dim
            att_output = att_output.reshape(batch_size, -1, 8 * head_dim)

            outputs_embeds = []
            start_pos = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]
                end_pos = start_pos + hidden_states.shape[1]

                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
                out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])
                after_first_residual = out_emb.clone()
                out_emb, gate = layer.post_attention_layernorm(out_emb, cond=conds[i])
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    out_emb = out_emb.to(dtype=torch.bfloat16)
                out_emb = layer.mlp(out_emb)
                out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)
                outputs_embeds.append(out_emb)
                start_pos = end_pos

            return outputs_embeds

        inputs_embeds = [prefix_embs, suffix_embs]
        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete,
                    layer_idx,
                    inputs_embeds,
                    att_2d_masks_4d,
                    position_ids,
                    conds,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                inputs_embeds = compute_layer_complete(layer_idx, inputs_embeds, att_2d_masks_4d, position_ids, conds)
            if layer_idx + 1 == self.wan_steering_block:
                inputs_embeds = [self.fuse_hidden_with_wan(inputs_embeds[0], wan_latents), inputs_embeds[1]]

        def compute_final_norms(inputs_embeds, conds):
            outputs_embeds = []
            for i, hidden_states in enumerate(inputs_embeds):
                out_emb, _ = models[i].norm(hidden_states, cond=conds[i])
                outputs_embeds.append(out_emb)
            return outputs_embeds

        if use_gradient_checkpointing:
            outputs_embeds = torch.utils.checkpoint.checkpoint(
                compute_final_norms,
                inputs_embeds,
                conds,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            outputs_embeds = compute_final_norms(inputs_embeds, conds)
        return outputs_embeds, None

    def forward(self, observation, actions, wan_latents=None, noise=None, time=None) -> Tensor:
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        if self.wan_steering_mode == "early":
            prefix_embs = self.fuse_prefix_with_wan(prefix_embs, wan_latents)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
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

        if self.wan_steering_mode == "block":
            (_, suffix_out), _ = self._forward_with_block_steering(
                prefix_embs,
                suffix_embs,
                att_2d_masks_4d,
                position_ids,
                adarms_cond,
                wan_latents,
            )
        else:
            def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
                (_, suffix_out), _ = self.paligemma_with_expert.forward(
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                )
                return suffix_out

            suffix_out = self._apply_checkpoint(
                forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
            )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, wan_latents=None, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        if self.wan_steering_mode == "early":
            prefix_embs = self.fuse_prefix_with_wan(prefix_embs, wan_latents)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        if self.wan_steering_mode == "block":
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                    state, x_t, expanded_time
                )
                if (
                    self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                    == torch.bfloat16
                ):
                    suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
                position_ids = torch.cumsum(pad_masks, dim=1) - 1
                att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
                (_, suffix_out), _ = self._forward_with_block_steering(
                    prefix_embs,
                    suffix_embs,
                    att_2d_masks_4d,
                    position_ids,
                    adarms_cond,
                    wan_latents,
                )
                suffix_out = suffix_out[:, -self.config.action_horizon :]
                suffix_out = suffix_out.to(dtype=torch.float32)
                v_t = self.action_out_proj(suffix_out)
                x_t = x_t + dt * v_t
                time += dt
            return x_t

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
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt
        return x_t
