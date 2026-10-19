"""schema=2 多任务 DT：分枝排序、独立停止判断、六维 ΔK 辅助预测。"""
from __future__ import annotations

import torch
import torch.nn as nn

from dt_model import DecisionTransformer, NEG_INF


class BranchStopKPIHead(nn.Module):
    def __init__(self, ctx_dim: int, node_dim: int, hidden: int = 128,
                 kpi_dim: int = 6, explicit_kpi_branch: bool = False,
                 monotonic_kpi_branch: bool = False,
                 use_kpi_branch: bool = True,
                 kpi_weights: tuple[float, ...] = (5, 10, 8, 8, 2, 5)):
        super().__init__()
        self.explicit_kpi_branch = explicit_kpi_branch
        self.monotonic_kpi_branch = monotonic_kpi_branch
        self.use_kpi_branch = use_kpi_branch
        self.node_proj = nn.Linear(node_dim, hidden)
        self.ctx_proj = nn.Linear(ctx_dim, hidden)
        branch_extra = 1 if explicit_kpi_branch and not monotonic_kpi_branch else 0
        self.branch_score = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden + branch_extra, hidden),
            nn.ReLU(), nn.Linear(hidden, 1))
        self.delta_k = nn.Sequential(
            nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, kpi_dim))
        weights = torch.as_tensor(kpi_weights, dtype=torch.float32)
        self.register_buffer("kpi_weights", weights / weights.abs().sum())
        self.net_gain = nn.Sequential(
            nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # ctx 已编码近期序列；再显式拼入最佳候选表示及其预测净收益。
        self.stop = nn.Sequential(
            nn.Linear(ctx_dim + hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def stop_features(self, ctx: torch.Tensor, nodes: torch.Tensor,
                      legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 Stop Head 输入及逐候选净收益，供在线监督复用。"""
        joint = self.node_proj(nodes) + self.ctx_proj(ctx).unsqueeze(1)
        net_gain = self.net_gain(joint).squeeze(-1).masked_fill(~legal, NEG_INF)
        best_idx = net_gain.argmax(dim=-1)
        rows = torch.arange(len(best_idx), device=ctx.device)
        best_joint = joint[rows, best_idx]
        best_net = net_gain[rows, best_idx].unsqueeze(-1)
        return torch.cat([ctx, best_joint, best_net], dim=-1), net_gain

    def forward(self, ctx: torch.Tensor, nodes: torch.Tensor,
                legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 branch_logits、stop_logit、delta_k 和逐候选预测净收益。"""
        joint = self.node_proj(nodes) + self.ctx_proj(ctx).unsqueeze(1)
        delta_k = self.delta_k(joint)
        utility = (delta_k * self.kpi_weights).sum(-1, keepdim=True)
        if self.monotonic_kpi_branch:
            base = self.branch_score(joint).squeeze(-1)
            branch = self._standardize_legal(base, legal)
            if self.use_kpi_branch:
                branch = branch + self._standardize_legal(
                    utility.squeeze(-1), legal)
        elif self.explicit_kpi_branch:
            if not self.use_kpi_branch:
                utility = torch.zeros_like(utility)
            branch_input = torch.cat([joint, utility], dim=-1)
            branch = self.branch_score(branch_input).squeeze(-1)
        else:
            branch = self.branch_score(joint).squeeze(-1)
        branch = branch.masked_fill(~legal, NEG_INF)
        stop_input, net_gain = self.stop_features(ctx, nodes, legal)
        stop = self.stop(stop_input).squeeze(-1)
        return branch, stop, delta_k, net_gain

    @staticmethod
    def _standardize_legal(values: torch.Tensor,
                           legal: torch.Tensor) -> torch.Tensor:
        """逐状态标准化候选分数，固定1:1融合，避免任一路径靠尺度吞没另一方。"""
        count = legal.sum(-1, keepdim=True).clamp_min(1)
        mean = values.masked_fill(~legal, 0.0).sum(-1, keepdim=True) / count
        centered = (values - mean).masked_fill(~legal, 0.0)
        var = centered.square().sum(-1, keepdim=True) / count
        return centered / torch.sqrt(var + 1e-6)


class DecisionTransformerV2(DecisionTransformer):
    def __init__(self, state_dim: int = 128, node_dim: int = 128,
                 hidden_dim: int = 128, n_layer: int = 3, n_head: int = 4,
                 dropout: float = 0.1, max_timestep: int = 1024,
                 kpi_dim: int = 6, explicit_kpi_branch: bool = False,
                 monotonic_kpi_branch: bool = False,
                 use_kpi_branch: bool = True,
                 kpi_weights: tuple[float, ...] = (5, 10, 8, 8, 2, 5)):
        super().__init__(
            state_dim=state_dim, node_dim=node_dim, hidden_dim=hidden_dim,
            n_layer=n_layer, n_head=n_head, dropout=dropout,
            max_timestep=max_timestep)
        self.head = BranchStopKPIHead(
            self.hidden_dim, node_dim, self.hidden_dim, kpi_dim,
            explicit_kpi_branch, monotonic_kpi_branch,
            use_kpi_branch, kpi_weights)
