from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from .fusion import WanFutureVideoFuser


class PI0WanLatentSteeringPytorch(PI0Pytorch):
    """pi0.5 PyTorch model with WorldPilot-style WAN latent steering."""

    def __init__(
        self,
        config,
        *,
        wan_num_heads: int = 8,
        wan_dropout: float = 0.0,
    ) -> None:
        super().__init__(config)
        hidden_dim = int(self.paligemma_with_expert.paligemma.config.text_config.hidden_size)
        self.wan_fuser = WanFutureVideoFuser(
            hidden_dim=hidden_dim,
            num_heads=wan_num_heads,
            dropout=wan_dropout,
        )

    def initialize_wan_fuser(self, wan_latents: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> None:
        """Materialize lazy fuser parameters before optimizer construction."""
        hidden_dim = int(self.paligemma_with_expert.paligemma.config.text_config.hidden_size)
        self.wan_fuser.to(device=device, dtype=dtype)
        dummy_hidden = torch.zeros(wan_latents.shape[0], 1, hidden_dim, device=device, dtype=dtype)
        self.wan_fuser(dummy_hidden, wan_latents.to(device=device, dtype=dtype), latent_layout="bvcthw")

    def fuse_prefix_with_wan(self, prefix_embs: Tensor, wan_latents: Tensor | None) -> Tensor:
        if wan_latents is None:
            return prefix_embs
        if wan_latents.ndim == 5:
            wan_latents = wan_latents.unsqueeze(0)
        return self.wan_fuser(
            prefix_embs,
            wan_latents.to(device=prefix_embs.device),
            latent_layout="bvcthw",
        ).hidden_states

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
        prefix_embs = self.fuse_prefix_with_wan(prefix_embs, wan_latents)

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
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt
        return x_t
