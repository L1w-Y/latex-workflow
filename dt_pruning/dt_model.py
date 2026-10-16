"""策略模型：分枝打分动作头 + Decision Transformer + 单步行为克隆（BC）。

动作空间按"当前树的分枝列表 + STOP"定义：动作头对每根分枝的向量打分，
STOP 由单独的输出头打分，输出 [G, N+1]（最后一列为 STOP）。分枝编号
只是树内序号、跨树无共同含义，因此不使用全局编号词表。

DT：GPT-2 骨干（L=3, H=4, d_h=128），输入 (RTG, state, action) 交错序列，
action token 用被剪分枝的向量（STOP 用可学习向量）表示，从 state token
位置预测本步动作。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

NEG_INF = -1e9
PAD_LABEL = -100              # 填充位标签（交叉熵 ignore_index）


def map_labels(labels: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """缓存标签 → 动作头列号：分枝局部序号不变，STOP(-1) → n_nodes，填充保持 PAD_LABEL。"""
    out = labels.clone()
    out[labels == -1] = n_nodes
    return out


class PointerHead(nn.Module):
    """对每根分枝打分（加性注意力）+ STOP 打分。"""

    def __init__(self, ctx_dim: int, node_dim: int, hidden: int = 128):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.ctx_proj = nn.Linear(ctx_dim, hidden)
        self.score = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, 1))
        self.stop = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, ctx: torch.Tensor, nodes: torch.Tensor,
                legal: torch.Tensor) -> torch.Tensor:
        """ctx [G,C]；nodes [G,N,D]；legal [G,N] bool → logits [G,N+1]。"""
        h = self.node_proj(nodes) + self.ctx_proj(ctx).unsqueeze(1)
        s = self.score(h).squeeze(-1).masked_fill(~legal, NEG_INF)
        return torch.cat([s, self.stop(ctx)], dim=-1)


class BCPolicy(nn.Module):
    """单步行为克隆：状态向量 → MLP → 分枝打分动作头。"""

    def __init__(self, state_dim: int = 128, node_dim: int = 128,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.ctx = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.head = PointerHead(hidden_dim, node_dim, hidden_dim)

    def forward(self, state, nodes, legal):
        return self.head(self.ctx(state), nodes, legal)


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim: int = 128, node_dim: int = 128,
                 hidden_dim: int = 128, n_layer: int = 3, n_head: int = 4,
                 dropout: float = 0.1, max_timestep: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_timestep = max_timestep

        config = GPT2Config(
            vocab_size=1, n_embd=hidden_dim, n_layer=n_layer, n_head=n_head,
            n_inner=hidden_dim * 4, activation_function="gelu",
            resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
            n_positions=3 * max_timestep + 3,
        )
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_timestep, hidden_dim)
        self.embed_rtg = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(node_dim, hidden_dim)
        self.stop_feat = nn.Parameter(torch.zeros(node_dim))
        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.head = PointerHead(hidden_dim, node_dim, hidden_dim)

    def action_features(self, nodes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """nodes [G,N,D]；labels [G]（已经 map_labels：0..N-1 分枝，N=STOP，<0 填充）→ [G,D]。"""
        G, N, _ = nodes.shape
        idx = labels.clamp(0, N - 1)
        feat = nodes[torch.arange(G, device=nodes.device), idx]
        feat = torch.where((labels == N).unsqueeze(-1),
                           self.stop_feat.expand_as(feat), feat)
        return feat * (labels >= 0).unsqueeze(-1).to(feat.dtype)

    def forward(self, states: torch.Tensor, act_feats: torch.Tensor,
                rtg: torch.Tensor, timesteps: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """states [B,T,S]；act_feats [B,T,D]；rtg [B,T,1]；timesteps [B,T]；
        attention_mask [B,T]。返回各 state 位置的上下文向量 [B,T,H]。"""
        B, T = states.shape[0], states.shape[1]
        time_emb = self.embed_timestep(timesteps.clamp(max=self.max_timestep - 1))
        h_rtg = self.embed_rtg(rtg) + time_emb
        h_state = self.embed_state(states) + time_emb
        h_action = self.embed_action(act_feats) + time_emb

        stacked = torch.stack([h_rtg, h_state, h_action], dim=2)
        stacked = self.embed_ln(stacked.reshape(B, 3 * T, self.hidden_dim))
        mask = (attention_mask.repeat_interleave(3, dim=1)
                if attention_mask is not None else None)
        out = self.transformer(inputs_embeds=stacked,
                               attention_mask=mask).last_hidden_state
        return out[:, 1::3, :]
