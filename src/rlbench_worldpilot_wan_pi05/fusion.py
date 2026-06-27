from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


LatentLayout = Literal["bvcthw", "bvtchw", "bcthw", "btchw"]
TimeMode = Literal["all", "last", "mean"]


@dataclass(frozen=True)
class WanFuserOutput:
    hidden_states: torch.Tensor
    latent_tokens: torch.Tensor


class WanFutureVideoFuser(nn.Module):
    """WorldPilot-style cross-attention fuser for WAN future video latents.

    Input VLM hidden states are kept at shape ``(B, L, H)``. WAN VAE-before-decode
    latents are converted into future-scene tokens and used as key/value tokens
    in a residual cross-attention block.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int = 8,
        max_views: int = 8,
        max_latent_steps: int = 64,
        time_mode: TimeMode = "all",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if time_mode not in {"all", "last", "mean"}:
            raise ValueError(f"Unsupported time_mode: {time_mode}")
        self.hidden_dim = hidden_dim
        self.time_mode = time_mode
        self.projector = nn.LazyLinear(hidden_dim)
        self.view_embed = nn.Embedding(max_views, hidden_dim)
        self.time_embed = nn.Embedding(max_latent_steps, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        vlm_hidden_states: torch.Tensor,
        wan_latents: torch.Tensor,
        *,
        latent_layout: LatentLayout = "bvcthw",
    ) -> WanFuserOutput:
        if vlm_hidden_states.ndim != 3:
            raise ValueError(f"vlm_hidden_states must be (B, L, H), got {tuple(vlm_hidden_states.shape)}")

        latents = self._to_bvcthw(wan_latents, latent_layout)
        flat, view_ids, time_ids = self._flatten_latents(latents)

        latent_tokens = self.projector(flat.to(dtype=vlm_hidden_states.dtype))
        latent_tokens = latent_tokens + self.view_embed(view_ids) + self.time_embed(time_ids)

        attn_out, _ = self.cross_attn(
            query=vlm_hidden_states,
            key=latent_tokens,
            value=latent_tokens,
            need_weights=False,
        )
        hidden_states = self.norm(vlm_hidden_states + self.dropout(attn_out))
        return WanFuserOutput(hidden_states=hidden_states, latent_tokens=latent_tokens)

    @staticmethod
    def _to_bvcthw(latents: torch.Tensor, layout: LatentLayout) -> torch.Tensor:
        if layout == "bvcthw":
            if latents.ndim != 6:
                raise ValueError(f"bvcthw expects 6 dims, got {tuple(latents.shape)}")
            return latents
        if layout == "bvtchw":
            if latents.ndim != 6:
                raise ValueError(f"bvtchw expects 6 dims, got {tuple(latents.shape)}")
            return latents.permute(0, 1, 3, 2, 4, 5).contiguous()
        if layout == "bcthw":
            if latents.ndim != 5:
                raise ValueError(f"bcthw expects 5 dims, got {tuple(latents.shape)}")
            return latents[:, None]
        if layout == "btchw":
            if latents.ndim != 5:
                raise ValueError(f"btchw expects 5 dims, got {tuple(latents.shape)}")
            return latents.permute(0, 2, 1, 3, 4).contiguous()[:, None]
        raise ValueError(f"Unsupported latent_layout: {layout}")

    def _flatten_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_views, channels, latent_steps, height, width = latents.shape
        device = latents.device

        if num_views > self.view_embed.num_embeddings:
            raise ValueError(f"num_views={num_views} exceeds max_views={self.view_embed.num_embeddings}")
        if latent_steps > self.time_embed.num_embeddings:
            raise ValueError(
                f"latent_steps={latent_steps} exceeds max_latent_steps={self.time_embed.num_embeddings}"
            )

        if self.time_mode == "all":
            tokens = latents.permute(0, 1, 3, 2, 4, 5).reshape(
                bsz, num_views * latent_steps, channels * height * width
            )
            view_ids = torch.arange(num_views, device=device).repeat_interleave(latent_steps)
            time_ids = torch.arange(latent_steps, device=device).repeat(num_views)
        elif self.time_mode == "last":
            tokens = latents[:, :, :, -1].reshape(bsz, num_views, channels * height * width)
            view_ids = torch.arange(num_views, device=device)
            time_ids = torch.full((num_views,), latent_steps - 1, device=device, dtype=torch.long)
        else:
            tokens = latents.mean(dim=3).reshape(bsz, num_views, channels * height * width)
            view_ids = torch.arange(num_views, device=device)
            time_ids = torch.zeros(num_views, device=device, dtype=torch.long)

        view_ids = view_ids[None].expand(bsz, -1)
        time_ids = time_ids[None].expand(bsz, -1)
        return tokens, view_ids, time_ids

