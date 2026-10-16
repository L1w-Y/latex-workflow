"""基线策略：贪心几何规则（论文 eq:greedy_score）、专家规则、随机。"""
from __future__ import annotations

import numpy as np

from config import CFG
from reward import (BranchProps, RewardConfig, detect_crossing)
from trajectories import expert_action
from tree_data import TreeState


def greedy_action(state: TreeState, props: dict[int, BranchProps],
                  tree_radius: float, cfg: RewardConfig | None = None
                  ) -> int | None:
    """贪心基线（论文 eq:greedy_score / eq:greedy_total / eq:greedy_action）。

    s1 内膛倾向、s2 竖直度、s3 长枝拥挤倾向；最高分 < τ_g 时停止。
    """
    ecfg = CFG.eval
    legal = state.legal_actions(CFG.data.min_nodes_abs, CFG.data.min_nodes_ratio)
    if not legal:
        return None
    trunk_xy = state.skeleton[np.argmin(state.skeleton[:, 2])][:2]

    cands = [props[b] for b in legal if b in props]
    if not cands:
        return None
    l_star = max(p.length for p in cands) + 1e-9
    rho = {p.branch_id: float(np.linalg.norm(p.center[:2] - trunk_xy))
           for p in cands}
    rho_star = max(rho.values()) + 1e-9
    alpha0 = ecfg.greedy_alpha0

    best_bid, best_score = None, -np.inf
    for p in cands:
        alpha_b = float(np.rad2deg(np.arccos(
            np.clip(p.vector[2], -1.0, 1.0))))
        s1 = 1.0 - min(rho[p.branch_id] / (tree_radius + 1e-9), 1.0)
        s2 = max(1.0 - alpha_b / alpha0, 0.0)
        s3 = (p.length / l_star) * (1.0 - rho[p.branch_id] / rho_star)
        w1, w2, w3 = ecfg.greedy_weights
        score = w1 * s1 + w2 * s2 + w3 * s3
        if score > best_score:
            best_bid, best_score = p.branch_id, score
    if best_score < ecfg.greedy_threshold:
        return None
    return best_bid


def tree_horizontal_radius(sk: np.ndarray) -> float:
    """R_t：树水平尺度的常数估计（初始树包围盒水平对角线的一半）。"""
    xy = sk[:, :2]
    return float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)) / 2.0)


__all__ = ["greedy_action", "tree_horizontal_radius", "expert_action"]
