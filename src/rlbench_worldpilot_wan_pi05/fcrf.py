from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

LatentLayout = Literal["bvcthw", "bvtchw", "bcthw", "btchw"]


@dataclass(frozen=True)
class FCRFResidualOutput:
    delta_flow: torch.Tensor
    gate: torch.Tensor
    correction: torch.Tensor
    latent_tokens: torch.Tensor


class WanFutureTokenEncoder(nn.Module):
    """Encode one token for every (view, latent-time) pair.

    The default three-view, six-latent-frame cache therefore produces 18
    temporally and view-positioned tokens. This is a fresh FCRF encoder; it
    does not load the GFPI prefix-fuser weights.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        max_views: int = 8,
        max_latent_steps: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.projector = nn.LazyLinear(self.hidden_dim)
        self.view_embed = nn.Embedding(int(max_views), self.hidden_dim)
        self.time_embed = nn.Embedding(int(max_latent_steps), self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        wan_latents: torch.Tensor,
        *,
        latent_layout: LatentLayout = "bvcthw",
    ) -> torch.Tensor:
        latents = self._to_bvcthw(wan_latents, latent_layout)
        flat, view_ids, time_ids = self._flatten_latents(latents)
        compute_dtype = self.view_embed.weight.dtype
        tokens = self.projector(flat.to(dtype=compute_dtype))
        tokens = tokens + self.view_embed(view_ids) + self.time_embed(time_ids)
        return self.norm(tokens)

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

    def _flatten_latents(
        self,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_views, channels, latent_steps, height, width = latents.shape
        if num_views > self.view_embed.num_embeddings:
            raise ValueError(f"num_views={num_views} exceeds max_views={self.view_embed.num_embeddings}")
        if latent_steps > self.time_embed.num_embeddings:
            raise ValueError(f"latent_steps={latent_steps} exceeds max_latent_steps={self.time_embed.num_embeddings}")

        flat = latents.permute(0, 1, 3, 2, 4, 5).reshape(
            bsz,
            num_views * latent_steps,
            channels * height * width,
        )
        device = latents.device
        view_ids = torch.arange(num_views, device=device).repeat_interleave(latent_steps)
        time_ids = torch.arange(latent_steps, device=device).repeat(num_views)
        return (
            flat,
            view_ids[None].expand(bsz, -1),
            time_ids[None].expand(bsz, -1),
        )


class FCRFResidualFlow(nn.Module):
    """WAN-conditioned, sample-gated residual on a frozen action flow field."""

    def __init__(
        self,
        action_hidden_dim: int,
        action_dim: int,
        *,
        num_heads: int = 8,
        dropout: float = 0.0,
        gate_bias: float = -2.2,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        hidden_dim = int(action_hidden_dim)
        self.action_hidden_dim = hidden_dim
        self.action_dim = int(action_dim)
        self.token_encoder = WanFutureTokenEncoder(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.flow_time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.residual_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * int(mlp_ratio)),
            nn.SiLU(),
            nn.Linear(hidden_dim * int(mlp_ratio), hidden_dim),
        )
        self.residual_out = nn.Linear(hidden_dim, self.action_dim)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Exact base equivalence at initialization.
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, float(gate_bias))

    def forward(
        self,
        suffix_out_base: torch.Tensor,
        wan_latents: torch.Tensor,
        flow_time: torch.Tensor,
        *,
        latent_layout: LatentLayout = "bvcthw",
    ) -> FCRFResidualOutput:
        if suffix_out_base.ndim != 3:
            raise ValueError(f"suffix_out_base must be (B, action_horizon, H), got {tuple(suffix_out_base.shape)}")
        if flow_time.ndim != 1 or flow_time.shape[0] != suffix_out_base.shape[0]:
            raise ValueError(
                f"flow_time must be (B,), got {tuple(flow_time.shape)} for batch={suffix_out_base.shape[0]}"
            )

        compute_dtype = self.query_norm.weight.dtype
        query = self.query_norm(suffix_out_base.to(dtype=compute_dtype))
        time_hidden = self.flow_time_mlp(flow_time[:, None].to(dtype=compute_dtype))
        query_with_time = query + time_hidden[:, None, :]
        latent_tokens = self.token_encoder(wan_latents, latent_layout=latent_layout)
        attended, _ = self.cross_attn(
            query=query_with_time,
            key=latent_tokens,
            value=latent_tokens,
            need_weights=False,
        )
        attended = self.dropout(attended)
        residual_hidden = query_with_time + attended
        residual_hidden = residual_hidden + self.residual_mlp(residual_hidden)
        delta_flow = self.residual_out(residual_hidden).to(dtype=torch.float32)

        gate_features = torch.cat(
            [
                query_with_time.mean(dim=1),
                attended.mean(dim=1),
                time_hidden,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_mlp(gate_features).to(dtype=torch.float32))[:, None, :]
        correction = gate * delta_flow
        return FCRFResidualOutput(
            delta_flow=delta_flow,
            gate=gate,
            correction=correction,
            latent_tokens=latent_tokens,
        )
