from typing import Optional

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    Dual-branch state encoder.

    Inputs:
    - point_cloud: [B, N, 3]
    - skeleton:    [B, M, 6]

    Output:
    - state:       [B, out_dim]
    """

    def __init__(self, hidden_dim: int = 128, out_dim: int = 128) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.skel_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, point_cloud: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        if point_cloud.ndim != 3 or point_cloud.size(-1) != 3:
            raise ValueError(f"point_cloud must be [B, N, 3], got {tuple(point_cloud.shape)}")
        if skeleton.ndim != 3 or skeleton.size(-1) != 6:
            raise ValueError(f"skeleton must be [B, M, 6], got {tuple(skeleton.shape)}")

        if point_cloud.size(1) == 0 or skeleton.size(1) == 0:
            raise ValueError("point_cloud and skeleton must contain at least one point/node")

        p_feat = self.point_mlp(point_cloud)          # [B, N, 64]
        s_feat = self.skel_mlp(skeleton)              # [B, M, 64]

        p_global = p_feat.max(dim=1).values           # [B, 64]
        s_global = s_feat.max(dim=1).values           # [B, 64]

        fused = torch.cat([p_global, s_global], dim=-1)  # [B, 128]
        state = self.fusion_mlp(fused)                    # [B, 64]
        return state


class PruningDecisionTransformer(nn.Module):
    """
    GPT-style causal transformer for offline RL sequence modeling.

    Token pattern per timestep t: [R_t, s_t, a_t]

    Inputs:
    - returns_to_go: [B, T, 1]
    - states:        [B, T, state_dim]
    - actions:       [B, T] (discrete action ids)

    Output:
    - action_logits: [B, T, action_dim]
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int = 128,
        d_model: int = 256,
        n_layer: int = 3,
        n_head: int = 4,
        dropout: float = 0.1,
        max_timestep: int = 256,
    ) -> None:
        super().__init__()

        if action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if state_dim <= 0:
            raise ValueError("state_dim must be > 0")

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.d_model = d_model
        self.max_seq_tokens = max_timestep * 3

        self.state_embed = nn.Linear(state_dim, d_model)
        self.return_embed = nn.Linear(1, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)
        self.pos_embed = nn.Embedding(self.max_seq_tokens, d_model)
        self.embed_ln = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.action_head = nn.Linear(d_model, action_dim)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # True means masked in nn.TransformerEncoder
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        returns_to_go: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if returns_to_go.ndim != 3 or returns_to_go.size(-1) != 1:
            raise ValueError(f"returns_to_go must be [B, T, 1], got {tuple(returns_to_go.shape)}")
        if states.ndim != 3 or states.size(-1) != self.state_dim:
            raise ValueError(f"states must be [B, T, {self.state_dim}], got {tuple(states.shape)}")
        if actions.ndim != 2:
            raise ValueError(f"actions must be [B, T], got {tuple(actions.shape)}")

        bsz, t, _ = states.shape
        if returns_to_go.shape[:2] != (bsz, t) or actions.shape[:2] != (bsz, t):
            raise ValueError("Batch/sequence dimensions of inputs are inconsistent")

        if t * 3 > self.max_seq_tokens:
            raise ValueError(
                f"Sequence too long: 3*T={t*3} exceeds max token length {self.max_seq_tokens}. "
                "Increase max_timestep in constructor."
            )

        device = states.device
        actions = actions.long().to(device)

        r_tok = self.return_embed(returns_to_go)      # [B, T, 128]
        s_tok = self.state_embed(states)              # [B, T, 128]
        a_tok = self.action_embed(actions)            # [B, T, 128]

        # Interleave [R_t, s_t, a_t]
        tokens = torch.stack([r_tok, s_tok, a_tok], dim=2)  # [B, T, 3, 128]
        tokens = tokens.reshape(bsz, t * 3, self.d_model)   # [B, 3T, 128]

        pos_ids = torch.arange(t * 3, device=device).unsqueeze(0).expand(bsz, -1)
        x = tokens + self.pos_embed(pos_ids)
        x = self.embed_ln(x)
        x = self.embed_drop(x)

        causal_mask = self._causal_mask(t * 3, device)

        if key_padding_mask is not None:
            if key_padding_mask.shape != (bsz, t):
                raise ValueError("key_padding_mask must be [B, T]")
            # Expand per timestep mask into token mask for [R,s,a]
            kpm = key_padding_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(bsz, t * 3)
        else:
            kpm = None

        h = self.transformer(x, mask=causal_mask, src_key_padding_mask=kpm)  # [B, 3T, 128]

        # Predict action logits from state token positions (index 1,4,7,...)
        state_h = h[:, 1::3, :]                          # [B, T, 128]
        action_logits = self.action_head(state_h)        # [B, T, action_dim]
        return action_logits


__all__ = ["StateEncoder", "PruningDecisionTransformer"]
