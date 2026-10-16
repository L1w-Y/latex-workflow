"""离线强化学习基线（离散动作版）：CQL / BCQ / IQL。

与 DT、BC 共用分支级图编码器与分枝打分头（PointerHead），动作空间同为
"当前树每根分枝 + STOP"；Q 值即打分头对各动作的输出。

- CQL（Kumar et al., 2020）：TD 损失 + α·(logsumexp_a Q(s,a) − Q(s,a_data))，压低数据外动作的 Q 值
- BCQ（Fujimoto et al., 2019，离散版）：另训模仿头 G(a|s)，只在 G(a|s)/max G ≥ τ 的动作中取 Q 最大
- IQL（Kostrikov et al., 2022）：V 用 Q 在数据动作上的 expectile 回归，Q 用 r + γV(s') 拟合，
  策略按 exp(β·(Q − V)) 加权模仿数据动作；全程不查询数据外动作的 Q 值
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dt_model import NEG_INF, BCPolicy

ALGOS = ("cql", "bcq", "iql")


class OfflineAgent(nn.Module):
    def __init__(self, algo: str, state_dim: int = 128, node_dim: int = 128,
                 hidden_dim: int = 128, dropout: float = 0.1, bcq_threshold: float = 0.3):
        super().__init__()
        if algo not in ALGOS:
            raise ValueError(algo)
        self.algo = algo
        self.bcq_log_tau = math.log(bcq_threshold)
        self.q = BCPolicy(state_dim, node_dim, hidden_dim, dropout)
        if algo == "bcq":
            self.imit = BCPolicy(state_dim, node_dim, hidden_dim, dropout)
        if algo == "iql":
            self.v = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, 1))
            self.pi = BCPolicy(state_dim, node_dim, hidden_dim, dropout)

    def bcq_allowed(self, imit_logits: torch.Tensor) -> torch.Tensor:
        lp = F.log_softmax(imit_logits, dim=-1)
        return lp - lp.max(dim=-1, keepdim=True).values >= self.bcq_log_tau

    def forward(self, state, nodes, legal) -> torch.Tensor:
        """推理用动作打分 [G, N+1]（取 argmax 即动作）。"""
        if self.algo == "cql":
            return self.q(state, nodes, legal)
        if self.algo == "bcq":
            q = self.q(state, nodes, legal)
            return q.masked_fill(~self.bcq_allowed(self.imit(state, nodes, legal)), NEG_INF)
        return self.pi(state, nodes, legal)


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    w = torch.where(diff > 0, tau, 1 - tau)
    return (w * diff.pow(2)).mean()


def compute_loss(agent: OfflineAgent, target: OfflineAgent, enc_out, enc_next_t, enc_cur_t,
                 labels, reward, done, hp: dict) -> tuple[torch.Tensor, dict]:
    """enc_out：在线编码器对 s 的输出；enc_next_t：目标编码器对 s' 的输出；
    enc_cur_t：目标编码器对 s 的输出（仅 IQL 用）。labels 已 map_labels。"""
    state, nodes, legal = enc_out
    s2, n2, l2 = enc_next_t
    G = labels.size(0)
    idx = torch.arange(G, device=labels.device)
    gamma = hp["gamma"]
    q = agent.q(state, nodes, legal)
    q_a = q[idx, labels]
    stats = {}

    if agent.algo in ("cql", "bcq"):
        with torch.no_grad():
            q2 = target.q(s2, n2, l2)
            if agent.algo == "bcq":
                q2 = q2.masked_fill(~target.bcq_allowed(target.imit(s2, n2, l2)), NEG_INF)
            y = reward + gamma * (1 - done) * q2.max(dim=-1).values
        td = F.smooth_l1_loss(q_a, y)
        loss = td
        stats["td"] = td.item()
        if agent.algo == "cql":
            cql = (torch.logsumexp(q, dim=-1) - q_a).mean()
            loss = loss + hp["cql_alpha"] * cql
            stats["cql"] = cql.item()
        else:
            ce = F.cross_entropy(agent.imit(state, nodes, legal), labels)
            loss = loss + ce
            stats["imit_ce"] = ce.item()
        stats["q_mean"] = q_a.mean().item()
        return loss, stats

    # IQL
    cs, cn, cl = enc_cur_t
    with torch.no_grad():
        q_t_a = target.q(cs, cn, cl)[idx, labels]
        v2 = target.v(s2).squeeze(-1)
        y = reward + gamma * (1 - done) * v2
    v = agent.v(state.detach()).squeeze(-1)
    v_loss = expectile_loss(q_t_a - v, hp["iql_tau"])
    q_loss = F.mse_loss(q_a, y)
    w = torch.exp(hp["iql_beta"] * (q_t_a - v.detach())).clamp(max=100.0)
    pi_logits = agent.pi(state.detach(), nodes.detach(), legal)
    pi_loss = (w * F.cross_entropy(pi_logits, labels, reduction="none")).mean()
    stats.update(q_loss=q_loss.item(), v_loss=v_loss.item(), pi_loss=pi_loss.item(),
                 q_mean=q_a.mean().item(), adv_w=w.mean().item())
    return q_loss + v_loss + pi_loss, stats


@torch.no_grad()
def polyak(target: nn.Module, online: nn.Module, tau: float):
    for pt, p in zip(target.parameters(), online.parameters()):
        pt.mul_(1 - tau).add_(p, alpha=tau)
