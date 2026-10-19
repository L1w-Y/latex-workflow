"""KPI 与复合奖励：全平台唯一实现（论文第 3 节口径）。

六个 KPI：层次性 H、拥挤度 C、平衡 B、光照 L、结果潜力 P、稳定性 S
复合奖励：R = α·R_q + β·R_i + γ·R_c

性能设计：
- 每个状态的枝条属性只计算一次（BranchProps 缓存），奖励与 KPI 共用
- 交叉枝检测用 KDTree 求最小距离，避免 O(n²m²) 双重循环
- 每棵树的 KPI 参照（主干基部、冠幅、分层高度、骨干枝、受光点）由 TreeContext 按初始树生成一次，
  之后各状态复用；光照用枝段加密球链 + 投影平面 KD 树查询
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from config import CFG, RewardConfig
from tree_data import Z, BRANCH, build_parent_map, trunk_branch_id

KPI_KEYS = ("hierarchy", "crowding", "balance", "light", "potential", "stability")


# ==============================================================================
# 枝条属性（论文 sec:formalization:properties）
# ==============================================================================

@dataclass
class BranchProps:
    branch_id: int
    points: np.ndarray        # [n, 3]
    vector: np.ndarray        # 单位生长方向 v_b
    start: np.ndarray         # 基部点
    end: np.ndarray           # 末端点
    center: np.ndarray        # 空间质心 c_b
    length: float             # 弧长 L_b
    avg_radius: float         # 平均半径 r̄_b
    avg_z: float
    parent_bid: int
    volume: float             # 估计体积 V_b = L·π·r̄²


def compute_branch_props(sk: np.ndarray, parent_map: dict[int, int],
                         eps: float = 1e-6) -> dict[int, BranchProps]:
    """一次性计算当前状态全部枝条属性。"""
    props: dict[int, BranchProps] = {}
    for bid in np.unique(sk[:, BRANCH].astype(np.int64)):
        rows = sk[sk[:, BRANCH] == bid]
        if rows.shape[0] < 2:
            continue
        pts = rows[:, :3]
        radii = rows[:, 3]
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        avg_r = float(np.mean(radii))
        direction = pts[-1] - pts[0]
        props[int(bid)] = BranchProps(
            branch_id=int(bid),
            points=pts,
            vector=direction / (np.linalg.norm(direction) + eps),
            start=pts[0], end=pts[-1],
            center=pts.mean(axis=0),
            length=length, avg_radius=avg_r,
            avg_z=float(pts[:, 2].mean()),
            parent_bid=parent_map.get(int(bid), -1),
            volume=length * np.pi * avg_r ** 2,
        )
    return props


def _angle_deg(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-6) -> float:
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + eps)
    return float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


def _effective_branches(props: dict[int, BranchProps], l_eff: float) -> list[BranchProps]:
    return [p for p in props.values() if p.length > l_eff]


def mean_length(props: dict[int, BranchProps], l_eff: float) -> float:
    eff = _effective_branches(props, l_eff)
    return float(np.mean([p.length for p in eff])) if eff else 0.0


def mean_radius(props: dict[int, BranchProps], l_eff: float) -> float:
    eff = _effective_branches(props, l_eff)
    return float(np.mean([p.avg_radius for p in eff])) if eff else 0.0


# ==============================================================================
# 不良枝判别（论文 sec:formalization:bad_branches）
# ==============================================================================

def detect_crossing(props: dict[int, BranchProps], cfg: RewardConfig
                    ) -> list[tuple[int, int, float, int]]:
    """四步级联筛选交叉/重叠枝对。

    返回 [(bid_a, bid_b, severity, worse_bid), ...]
    向量化优化：父子对排除 + 质心距离下界预筛用矩阵运算一次性完成，
    把 O(n²) 的 Python 循环缩减到少数候选对；后续精确判定保持原逻辑，
    结果与逐对循环严格一致（预筛加 1e-9 容差避免浮点误差漏判）。
    """
    bids = sorted(props.keys())
    n = len(bids)
    out = []
    if n < 2:
        return out

    centers = np.array([props[b].center for b in bids])
    lengths = np.array([props[b].length for b in bids])
    idx = {b: i for i, b in enumerate(bids)}

    # 父子对 mask（对称）
    parent_child = np.zeros((n, n), dtype=bool)
    for i, b in enumerate(bids):
        p = idx.get(props[b].parent_bid)
        if p is not None:
            parent_child[i, p] = True
            parent_child[p, i] = True

    # 质心距离下界矩阵：LB[i,j] = |c_i - c_j| - (L_i + L_j)/2，保守预筛
    D = squareform(pdist(centers))
    LB = D - (lengths[:, None] + lengths[None, :]) / 2.0
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    cand = upper & (LB <= cfg.d_cross + 1e-9) & ~parent_child
    ci, cj = np.nonzero(cand)

    # AABB 包围盒预筛：比质心下界更紧，进一步剔除候选对
    mins = np.stack([props[b].points.min(axis=0) for b in bids])
    maxs = np.stack([props[b].points.max(axis=0) for b in bids])
    seps = np.maximum(mins[ci] - maxs[cj], mins[cj] - maxs[ci])  # [M,3] 各轴分离
    keep = seps.max(axis=1) <= cfg.d_cross + 1e-9
    ci, cj = ci[keep], cj[keep]

    trees = {b: cKDTree(props[b].points) for b in bids}
    for i, j in zip(ci.tolist(), cj.tolist()):
        ba, bb = bids[i], bids[j]
        pa, pb = props[ba], props[bb]
        # 1) 空间邻近
        d_min = float(trees[ba].query(pb.points)[0].min())
        if d_min > cfg.d_cross:
            continue
        # 2) 非平行朝向
        theta = _angle_deg(pa.vector, pb.vector)
        if theta < cfg.theta_min or theta > cfg.theta_max:
            continue
        # 3) 水平投影重叠（论文 eq:overlap）
        sep = np.linalg.norm(pa.center[:2] - pb.center[:2])
        o_ij = max(0.0, 1.0 - sep / ((pa.length + pb.length) / 4.0 + cfg.eps))
        if o_ij < cfg.o_min:
            continue
        # 4) 竖直区间重叠
        za = (pa.points[:, 2].min(), pa.points[:, 2].max())
        zb = (pb.points[:, 2].min(), pb.points[:, 2].max())
        if za[1] < zb[0] or zb[1] < za[0]:
            continue
        severity = 0.5 * (1.0 - d_min / cfg.d_cross) + 0.5 * o_ij
        if severity <= cfg.sigma_min:
            continue
        out.append((ba, bb, severity, _inferior_branch(pa, pb, cfg)))
    return out


def _inferior_branch(pa: BranchProps, pb: BranchProps, cfg: RewardConfig) -> int:
    """劣质枝选择：半径、长度档、角度质量综合打分，低分者劣。"""
    def quality(p: BranchProps) -> float:
        score = 0.0
        score += 2.0 * p.avg_radius
        rel = p.length  # 与有利长度区间的匹配度
        if cfg.len_lambda1 <= rel <= cfg.len_lambda2:
            score += 3.0
        ang = _angle_deg(p.vector, np.array([0.0, 0.0, 1.0]))
        if 40.0 <= ang <= 80.0:
            score += 2.0
        return score
    return pa.branch_id if quality(pa) < quality(pb) else pb.branch_id


def is_inward(p: BranchProps, canopy_center: np.ndarray, cfg: RewardConfig) -> bool:
    """内膛枝（论文 eq:inward）。"""
    ref = p.start - canopy_center
    ref = ref / (np.linalg.norm(ref) + cfg.eps)
    return bool(np.dot(p.vector, ref) < cfg.tau_in and p.length > cfg.l_in)


def is_vigorous(p: BranchProps, avg_len: float, cfg: RewardConfig) -> bool:
    """徒长枝（论文 eq:vigorous_angle / eq:vigorous_length）。"""
    ang = _angle_deg(p.vector, np.array([0.0, 0.0, 1.0]))
    return bool(ang < cfg.alpha_vig and p.length > cfg.lambda_vig * avg_len)


def is_drooping(p: BranchProps, cfg: RewardConfig) -> bool:
    """下垂枝（论文 eq:drooping）。"""
    ang = _angle_deg(p.vector, np.array([0.0, 0.0, 1.0]))
    return bool(ang > cfg.alpha_drp)


def is_weak(p: BranchProps, avg_r: float, cfg: RewardConfig) -> bool:
    """细弱枝（论文 eq:weak_length / eq:weak_radius）。"""
    return bool(p.length > cfg.l_wk and p.avg_radius < cfg.kappa_wk * avg_r)


def is_medium_short(p: BranchProps, avg_len: float, cfg: RewardConfig) -> bool:
    """中短枝保护（论文 eq:ms_protect，相对树均长）。"""
    rel = p.length / (avg_len + cfg.eps)
    return bool(cfg.lambda_ms_lo <= rel <= cfg.lambda_ms_hi)


def is_sturdy(p: BranchProps, avg_r: float, cfg: RewardConfig) -> bool:
    """粗壮枝保护（论文 eq:sturdy_protect）。"""
    return bool(p.avg_radius > cfg.kappa_st * avg_r)


# ==============================================================================
# 六个 KPI（论文 sec:formalization:kpis；2026-10-12 修订口径，见工作进度 F5）
#
# 负载代理：非主干有效枝的枝段长度（枝叶与挂果长在细枝上），用于 H / B / S。
# 每棵树的参照量（主干基部、冠幅、分层高度、骨干枝、受光点）按初始树固定一次，
# 由 TreeContext 保存，之后各状态复用，保证剪前剪后差值只反映剪枝本身。
# ==============================================================================

def _load_segments(props: dict[int, BranchProps], trunk_bid: int, l_eff: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """非主干有效枝的枝段中点 [n, 3] 与枝段长度 [n]。"""
    mids, w = [], []
    for p in props.values():
        if p.branch_id == trunk_bid or p.length <= l_eff:
            continue
        mids.append(0.5 * (p.points[:-1] + p.points[1:]))
        w.append(np.linalg.norm(np.diff(p.points, axis=0), axis=1))
    if not mids:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(mids), np.concatenate(w)


def _scaffold_ids(props: dict[int, BranchProps], trunk_bid: int, cfg: RewardConfig) -> list[int]:
    return [b for b, p in props.items() if b != trunk_bid
            and p.length > cfg.main_branch_length and p.avg_radius > cfg.main_branch_radius]


def _limb_ratios(props: dict[int, BranchProps], scaffolds: list[int]) -> dict[int, float]:
    """骨干枝负载比 ρ_b = Σ_子树枝段 长度 × 到该骨干枝基部的水平距离 / r̄_b³。"""
    children: dict[int, list[int]] = {}
    for b, p in props.items():
        children.setdefault(p.parent_bid, []).append(b)
    out = {}
    for s in scaffolds:
        if s not in props:
            continue
        base = props[s].start[:2]
        stack, moment = [s], 0.0
        while stack:
            b = stack.pop()
            pts = props[b].points
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            mid = 0.5 * (pts[:-1, :2] + pts[1:, :2])
            moment += float((seg * np.linalg.norm(mid - base, axis=1)).sum())
            stack.extend(children.get(b, []))
        out[s] = moment / max(props[s].avg_radius, 1e-4) ** 3
    return out


def _light_directions(zenith_deg: float) -> np.ndarray:
    z = np.deg2rad(zenith_deg)
    side = [[np.sin(z) * np.cos(a), np.sin(z) * np.sin(a), np.cos(z)]
            for a in np.deg2rad([0.0, 90.0, 180.0, 270.0])]
    return np.array([[0.0, 0.0, 1.0]] + side)


class TreeContext:
    """每棵树的 KPI 参照，按初始树生成一次，各状态复用。"""

    def __init__(self, sk_initial: np.ndarray, cfg: RewardConfig, seed: int = 0):
        props = compute_branch_props(sk_initial, build_parent_map(sk_initial))
        self.trunk = trunk_branch_id(sk_initial)
        self.base = sk_initial[np.argmin(sk_initial[:, Z]), :3].copy()
        rest = sk_initial[sk_initial[:, BRANCH] != self.trunk]
        if rest.shape[0] == 0:
            rest = sk_initial
        self.z_mid = 0.5 * (float(np.percentile(rest[:, Z], 2)) + float(rest[:, Z].max()))
        self.spread = max(float(np.percentile(
            np.linalg.norm(rest[:, :2] - self.base[:2], axis=1), 90)), cfg.eps)

        self.scaffolds = _scaffold_ids(props, self.trunk, cfg)
        rho = list(_limb_ratios(props, self.scaffolds).values())
        self.rho0 = float(np.mean(rho)) if rho else 1.0

        pts, bids = [], []
        for p in props.values():
            if p.length > cfg.l_eff:
                pts.append(p.points)
                bids.append(np.full(p.points.shape[0], p.branch_id))
        pts = np.concatenate(pts) if pts else np.zeros((0, 3))
        bids = np.concatenate(bids) if bids else np.zeros(0, dtype=np.int64)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=min(cfg.light_n_targets, len(pts)), replace=False)
        self.targets, self.target_bids = pts[idx], bids[idx].astype(np.int64)
        self.dirs = _light_directions(cfg.light_zenith)

    def light(self, props: dict[int, BranchProps], cfg: RewardConfig) -> float:
        """内膛受光 L：受光点沿竖直与 4 个斜向向上，未被其他枝（半径 + 叶幕包络）遮挡的比例。

        枝段按不超过叶幕包络的间距加密为球链近似胶囊体，投影到垂直于光线的平面后用 KD 树查询。
        """
        alive = np.fromiter((b in props for b in self.target_bids), bool, len(self.target_bids))
        tp, tb = self.targets[alive], self.target_bids[alive]
        if len(tp) == 0:
            return 1.0
        occ, rad, ob = [], [], []
        step = cfg.light_r_leaf
        for p in props.values():
            a, b = p.points[:-1], p.points[1:]
            n = np.maximum(np.ceil(np.linalg.norm(b - a, axis=1) / step).astype(int), 1)
            t = np.concatenate([np.arange(k) / k for k in n])
            seg = np.repeat(np.arange(len(a)), n)
            occ.append(a[seg] + t[:, None] * (b - a)[seg])
            rad.append(np.full(len(seg), p.avg_radius + cfg.light_r_leaf))
            ob.append(np.full(len(seg), p.branch_id))
            occ.append(p.points[-1:]); rad.append(rad[-1][:1]); ob.append(ob[-1][:1])
        occ, rad, ob = np.concatenate(occ), np.concatenate(rad), np.concatenate(ob)
        r_max = float(rad.max())
        lit = np.ones((len(tp), len(self.dirs)), dtype=bool)
        for k, d in enumerate(self.dirs):
            u = np.cross(d, [1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.cross(d, [0.0, 1.0, 0.0])
            u /= np.linalg.norm(u)
            v = np.cross(d, u)
            o2 = np.stack([occ @ u, occ @ v], axis=1)
            t2 = np.stack([tp @ u, tp @ v], axis=1)
            o_t, t_t = occ @ d, tp @ d
            kd = cKDTree(o2)
            for i, cand in enumerate(kd.query_ball_point(t2, r_max)):
                if not cand:
                    continue
                c = np.asarray(cand)
                d2 = ((o2[c] - t2[i]) ** 2).sum(axis=1)
                hit = (d2 <= rad[c] ** 2) & (o_t[c] > t_t[i] + 1e-6) & (ob[c] != tb[i])
                lit[i, k] = not hit.any()
        return float(lit.mean())


def kpi_hierarchy(props: dict[int, BranchProps], cfg: RewardConfig, ctx: TreeContext) -> float:
    """结构层次性 H = w_layer·H_layer + (1 − w_layer)·H_main。

    H_layer：冠层下半部（初始树分层高度以下）负载占比，"下大上小"越明显越高；
    H_main：骨干枝按高度排序后长度、半径逐级递减的比例（论文 eq:hierarchy_main）。
    """
    mids, w = _load_segments(props, ctx.trunk, cfg.l_eff)
    h_layer = float(w[mids[:, 2] < ctx.z_mid].sum() / w.sum()) if w.sum() > 0 else 0.5
    mains = sorted(_scaffold_ids(props, ctx.trunk, cfg), key=lambda b: props[b].avg_z)
    if len(mains) < 2:
        h_main = 0.5
    else:
        m = len(mains)
        h_l = sum(props[mains[i]].length >= props[mains[i + 1]].length for i in range(m - 1)) / (m - 1)
        h_r = sum(props[mains[i]].avg_radius >= props[mains[i + 1]].avg_radius for i in range(m - 1)) / (m - 1)
        h_main = (h_l + h_r) / 2.0
    return float(cfg.w_layer * h_layer + (1.0 - cfg.w_layer) * h_main)


def kpi_crowding(props: dict[int, BranchProps], cfg: RewardConfig) -> float:
    """局部拥挤度 C = 1 − 拥挤点占比（越大越通透）。

    有效枝上的骨架点，若 d_c 内存在其他枝的点即记为拥挤；分叉处（距任一方枝基部 < d_c）的近邻不计。
    """
    pts, bids, starts = [], [], []
    for p in props.values():
        if p.length <= cfg.l_eff:
            continue
        pts.append(p.points)
        bids.append(np.full(p.points.shape[0], p.branch_id))
        starts.append(np.repeat(p.start[None], p.points.shape[0], axis=0))
    if not pts:
        return 1.0
    pts, bids, starts = np.concatenate(pts), np.concatenate(bids), np.concatenate(starts)
    pairs = cKDTree(pts).query_pairs(cfg.crowd_radius, output_type="ndarray")
    if len(pairs) == 0:
        return 1.0
    i, j = pairs[:, 0], pairs[:, 1]
    near_fork = ((np.linalg.norm(pts[i] - starts[j], axis=1) < cfg.crowd_radius)
                 | (np.linalg.norm(pts[j] - starts[i], axis=1) < cfg.crowd_radius))
    ok = (bids[i] != bids[j]) & ~near_fork
    crowded = np.zeros(len(pts), dtype=bool)
    crowded[i[ok]] = True
    crowded[j[ok]] = True
    return float(1.0 - crowded.mean())


def kpi_balance(props: dict[int, BranchProps], cfg: RewardConfig, ctx: TreeContext) -> float:
    """树冠平衡 B = exp(−k_B·‖g_xy − t‖ / 冠幅)，g 为负载加权重心，t 为主干基部。"""
    mids, w = _load_segments(props, ctx.trunk, cfg.l_eff)
    if w.sum() <= 0:
        return 0.5
    g = (mids[:, :2] * w[:, None]).sum(axis=0) / w.sum()
    return float(np.exp(-cfg.k_b * np.linalg.norm(g - ctx.base[:2]) / ctx.spread))


def kpi_potential(props: dict[int, BranchProps], cfg: RewardConfig, trunk_bid: int) -> float:
    """结果潜力 P：有效非主干枝 P_b = f_θ·f_L 的均值（论文 eq:branch_potential；原为求和）。"""
    total, n = 0.0, 0
    for p in props.values():
        if p.branch_id == trunk_bid or p.length < cfg.l_eff:
            continue
        theta = float(np.rad2deg(np.arcsin(np.clip(p.vector[2], -1.0, 1.0))))
        if cfg.theta_fav[0] <= theta <= cfg.theta_fav[1]:
            f_theta = 1.0 + cfg.delta_fav
        elif theta > cfg.theta_vig:
            f_theta = 1.0 - cfg.delta_vig
        elif theta < cfg.theta_drp:
            f_theta = 1.0 - cfg.delta_drp
        else:
            f_theta = 1.0
        if cfg.len_lambda1 <= p.length <= cfg.len_lambda2:
            f_len = 1.0 + cfg.delta_mid
        elif p.length > cfg.len_lambda3:
            f_len = 1.0 - cfg.delta_long
        else:
            f_len = 1.0
        total += f_theta * f_len
        n += 1
    return float(total / n) if n else 0.0


def kpi_stability(props: dict[int, BranchProps], ctx: TreeContext) -> float:
    """结构稳定性 S = mean_b exp(−ρ_b / ρ̄_0)：骨干枝负载力矩相对其抗弯能力（∝ r̄³）。

    骨干枝按初始树固定，ρ̄_0 为该树剪前均值；已被剪掉的骨干枝不计入均值。
    """
    rho = _limb_ratios(props, ctx.scaffolds)
    if not rho:
        return 1.0
    return float(np.mean([np.exp(-r / ctx.rho0) for r in rho.values()]))


def compute_kpis(sk: np.ndarray, props: dict[int, BranchProps],
                 cfg: RewardConfig, ctx: TreeContext | None = None
                 ) -> dict[str, float]:
    """计算全部 KPI。ctx 为 None 时按当前骨架新建（仅用于一次性评估）。"""
    if sk.shape[0] == 0:
        return {"branch_count": 0.0, "hierarchy": 0.5, "crowding": 1.0,
                "balance": 0.5, "light": 1.0, "potential": 0.0, "stability": 1.0}
    if ctx is None:
        ctx = TreeContext(sk, cfg)
    return {
        "branch_count": float(len(props)),
        "hierarchy": kpi_hierarchy(props, cfg, ctx),
        "crowding": kpi_crowding(props, cfg),
        "balance": kpi_balance(props, cfg, ctx),
        "light": ctx.light(props, cfg),
        "potential": kpi_potential(props, cfg, ctx.trunk),
        "stability": kpi_stability(props, ctx),
    }


LightContext = TreeContext


# ==============================================================================
# 复合奖励（论文 sec:formalization:reward）
# ==============================================================================

def stop_reward(pruning_ratio: float, cfg: RewardConfig) -> float:
    """终止动作奖励（论文 eq:stop_reward）。"""
    eta = pruning_ratio
    if cfg.eta_lo <= eta <= cfg.eta_hi:
        return cfg.r_ideal
    if cfg.eta_buf_lo <= eta < cfg.eta_lo or cfg.eta_hi < eta <= cfg.eta_buf_hi:
        return cfg.r_buf
    if eta < cfg.eta_under:
        return -cfg.r_under
    if eta > cfg.eta_over:
        return -cfg.r_over
    return 0.0


def branch_quality_reward(action_bid: int,
                          props_before: dict[int, BranchProps],
                          crossing: list[tuple[int, int, float, int]],
                          canopy_center: np.ndarray,
                          avg_len: float, avg_r: float,
                          cfg: RewardConfig) -> float:
    """枝条质量奖励 R_q（论文 tab:quality_reward）。

    不良特征奖励可叠加；功能枝保护仅在无任何不良特征时生效。
    """
    p = props_before.get(action_bid)
    if p is None:
        return 0.0
    rq = 0.0
    for ba, bb, sev, worse in crossing:
        if action_bid == worse:
            rq += cfg.omega_cross_inferior * sev
            break
        if action_bid in (ba, bb):
            rq += cfg.omega_cross_other * sev
            break
    if is_inward(p, canopy_center, cfg):
        rq += cfg.omega_in
    if is_vigorous(p, avg_len, cfg):
        rq += cfg.omega_vig
    if is_drooping(p, cfg):
        rq += cfg.omega_drp
    if is_weak(p, avg_r, cfg):
        rq += cfg.omega_wk
    if rq == 0.0:
        if is_medium_short(p, avg_len, cfg):
            rq -= cfg.pi_ms
        if is_sturdy(p, avg_r, cfg):
            rq -= cfg.pi_st
    return float(rq)


def branch_tags(action_bid: int,
                props_before: dict[int, BranchProps],
                crossing: list[tuple[int, int, float, int]],
                canopy_center: np.ndarray,
                avg_len: float, avg_r: float,
                cfg: RewardConfig) -> list[str]:
    """被剪枝条的类别标签，判别顺序与 branch_quality_reward 一致（只做标注，不参与奖励）。"""
    p = props_before.get(action_bid)
    if p is None:
        return []
    tags = []
    for ba, bb, _, worse in crossing:
        if action_bid == worse:
            tags.append("cross_inferior")
            break
        if action_bid in (ba, bb):
            tags.append("cross_other")
            break
    if is_inward(p, canopy_center, cfg):
        tags.append("inward")
    if is_vigorous(p, avg_len, cfg):
        tags.append("vigorous")
    if is_drooping(p, cfg):
        tags.append("drooping")
    if is_weak(p, avg_r, cfg):
        tags.append("weak")
    if not tags:
        if is_medium_short(p, avg_len, cfg):
            tags.append("protect_medium_short")
        if is_sturdy(p, avg_r, cfg):
            tags.append("protect_sturdy")
    if p.length > cfg.main_branch_length and p.avg_radius > cfg.main_branch_radius:
        tags.append("main_branch")
    return tags


def branch_context(sk: np.ndarray, props: dict[int, BranchProps],
                   cfg: RewardConfig) -> tuple[np.ndarray, float, float]:
    """枝条判别所需的整树上下文：冠层中心、有效枝均长、有效枝均径。"""
    return sk[:, :3].mean(axis=0), mean_length(props, cfg.l_eff), mean_radius(props, cfg.l_eff)


def state_improvement_reward(k_before: dict[str, float], k_after: dict[str, float],
                             cfg: RewardConfig) -> float:
    """状态改善奖励 R_i = Σ_k W_k·ΔK_k / Δe_k，六项均为越高越好；不逐刀截断，轨迹求和与切分方式无关。"""
    return float(sum(cfg.kpi_weights[k] * (k_after[k] - k_before[k]) / cfg.kpi_scales[k] for k in KPI_KEYS))


def constraint_penalty(action_bid: int, props_before: dict[int, BranchProps],
                       cfg: RewardConfig) -> float:
    """约束惩罚 R_c：骨干枝保护 + 单步体积约束（论文 eq:constraint）。"""
    p = props_before.get(action_bid)
    if p is None:
        return 0.0
    rc = 0.0
    if p.length > cfg.main_branch_length and p.avg_radius > cfg.main_branch_radius:
        rc -= cfg.pi_main
    total_v = sum(q.volume for q in props_before.values())
    if total_v > 1e-12:
        ratio = p.volume / total_v
        if ratio > cfg.eta_v:
            rc -= (ratio - cfg.eta_v) * cfg.lambda_v
    return float(rc)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -40.0, 40.0))
    return 1.0 / (1.0 + np.exp(-x))


def continuous_constraint_penalty(action_bid: int,
                                  props_before: dict[int, BranchProps],
                                  cfg: RewardConfig) -> float:
    """v2 连续安全风险。

    长度和半径越接近骨干枝阈值，风险平滑增大；剪除体积占比在全区间
    都有信号，超过 ``eta_v`` 后按平方项加速惩罚。
    """
    p = props_before.get(action_bid)
    if p is None:
        return 0.0
    softness = max(cfg.v2_main_softness, cfg.eps)
    length_risk = _sigmoid((p.length / max(cfg.main_branch_length, cfg.eps) - 1.0) / softness)
    radius_risk = _sigmoid((p.avg_radius / max(cfg.main_branch_radius, cfg.eps) - 1.0) / softness)
    main_risk = cfg.pi_main * length_risk * radius_risk
    total_v = sum(q.volume for q in props_before.values())
    volume_ratio = p.volume / max(total_v, cfg.eps)
    rel = volume_ratio / max(cfg.eta_v, cfg.eps)
    volume_risk = cfg.v2_volume_risk_weight * (rel + max(rel - 1.0, 0.0) ** 2)
    return float(-(main_risk + volume_risk))


def pruning_cost(action_bid: int, sk_before: np.ndarray, sk_after: np.ndarray,
                 props_before: dict[int, BranchProps], cfg: RewardConfig) -> float:
    """v2 归一化剪除成本；兼顾枝体积和级联删除的骨架点比例。"""
    p = props_before.get(action_bid)
    if p is None:
        return 0.0
    total_v = sum(q.volume for q in props_before.values())
    volume_ratio = p.volume / max(total_v, cfg.eps)
    point_ratio = max(sk_before.shape[0] - sk_after.shape[0], 0) / max(sk_before.shape[0], 1)
    mix = float(np.clip(cfg.v2_prune_cost_volume_mix, 0.0, 1.0))
    variable = cfg.v2_prune_cost_weight * (
        mix * volume_ratio + (1.0 - mix) * point_ratio)
    return float(cfg.v2_fixed_action_cost + variable)


V2_RULE_FEATURE_KEYS = (
    "cross_inferior", "cross_other", "inward", "vigorous", "drooping", "weak",
    "protect_medium_short", "protect_sturdy", "main_risk", "volume_risk",
)


def branch_rule_features(props: dict[int, BranchProps],
                         crossing: list[tuple[int, int, float, int]],
                         canopy_center: np.ndarray, avg_len: float, avg_r: float,
                         cfg: RewardConfig) -> dict[int, np.ndarray]:
    """返回与园艺规则同源的逐枝显式特征，供 schema=2 编码器使用。"""
    cross_bad: dict[int, float] = {}
    cross_other: dict[int, float] = {}
    for ba, bb, severity, worse in crossing:
        cross_bad[worse] = max(cross_bad.get(worse, 0.0), float(severity))
        other = bb if worse == ba else ba
        cross_other[other] = max(cross_other.get(other, 0.0), float(severity))
    total_v = sum(q.volume for q in props.values())
    out = {}
    for bid, p in props.items():
        softness = max(cfg.v2_main_softness, cfg.eps)
        main_risk = _sigmoid((p.length / max(cfg.main_branch_length, cfg.eps) - 1.0) / softness)
        main_risk *= _sigmoid((p.avg_radius / max(cfg.main_branch_radius, cfg.eps) - 1.0) / softness)
        out[bid] = np.asarray([
            cross_bad.get(bid, 0.0),
            cross_other.get(bid, 0.0),
            float(is_inward(p, canopy_center, cfg)),
            float(is_vigorous(p, avg_len, cfg)),
            float(is_drooping(p, cfg)),
            float(is_weak(p, avg_r, cfg)),
            float(is_medium_short(p, avg_len, cfg)),
            float(is_sturdy(p, avg_r, cfg)),
            main_risk,
            p.volume / max(total_v, cfg.eps),
        ], dtype=np.float32)
    return out


def composite_reward_v2(action_bid: int,
                        sk_before: np.ndarray, props_before: dict[int, BranchProps],
                        k_before: dict[str, float],
                        sk_after: np.ndarray, props_after: dict[int, BranchProps],
                        k_after: dict[str, float],
                        crossing: list[tuple[int, int, float, int]],
                        cfg: RewardConfig) -> tuple[float, dict[str, float]]:
    """v2 净收益：规则质量 + KPI 改善 + 连续安全约束 − 剪除成本。"""
    canopy_center, avg_len, avg_r = branch_context(sk_before, props_before, cfg)
    rq = branch_quality_reward(action_bid, props_before, crossing,
                               canopy_center, avg_len, avg_r, cfg)
    if cfg.rq_by_volume:
        rq *= (sk_before.shape[0] - sk_after.shape[0]) / cfg.rq_volume_ref
    ri = state_improvement_reward(k_before, k_after, cfg)
    rc = continuous_constraint_penalty(action_bid, props_before, cfg)
    cost = pruning_cost(action_bid, sk_before, sk_after, props_before, cfg)
    total = float(cfg.alpha * rq + cfg.beta * ri + cfg.gamma * rc - cost)
    return total, {"rq": rq, "ri": ri, "rc": rc, "cost": cost}


def composite_reward(action_bid: int,
                     sk_before: np.ndarray, props_before: dict[int, BranchProps],
                     k_before: dict[str, float],
                     sk_after: np.ndarray, props_after: dict[int, BranchProps],
                     k_after: dict[str, float],
                     crossing: list[tuple[int, int, float, int]],
                     cfg: RewardConfig) -> tuple[float, dict[str, float]]:
    """复合奖励 R = α·R_q + β·R_i + γ·R_c。

    返回 (奖励值, 各分量明细)。
    """
    canopy_center, avg_len, avg_r = branch_context(sk_before, props_before, cfg)

    rq = branch_quality_reward(action_bid, props_before, crossing,
                               canopy_center, avg_len, avg_r, cfg)
    if cfg.rq_by_volume:
        rq *= (sk_before.shape[0] - sk_after.shape[0]) / cfg.rq_volume_ref
    ri = state_improvement_reward(k_before, k_after, cfg)
    rc = constraint_penalty(action_bid, props_before, cfg)

    total = float(cfg.alpha * rq + cfg.beta * ri + cfg.gamma * rc)
    return total, {"rq": rq, "ri": ri, "rc": rc}


# 与 reward_parts.step_rewards 的开关一致：不良枝得分和保护惩罚不同时非零，保护项是 rq 的负部。
# 四元组为：规则项、功能枝保护、指标项、约束项。
REWARD_SET_SWITCHES = {
    "full": (True, True, True, True),
    "norq": (False, False, True, True),
    "noprotect": (True, False, True, True),
    "nori": (True, True, False, True),
    "norc": (True, True, True, False),
}


def reward_for_set(parts: dict[str, float], reward_set: str,
                   cfg: RewardConfig | None = None) -> float:
    """推理时从目标回报中扣除的一步奖励。

    parts 来自 composite_reward：rq 已按剪除量缩放，ri 未乘 β，rc 未乘 γ。
    full 与 composite_reward 的合计相同；其余回报集只去掉对应分量。
    """
    if reward_set not in REWARD_SET_SWITCHES:
        raise ValueError(f"未知回报集 {reward_set}，可选 {tuple(REWARD_SET_SWITCHES)}")
    use_rq, use_protect, use_ri, use_rc = REWARD_SET_SWITCHES[reward_set]
    cfg = cfg or CFG.reward
    rq = parts["rq"]
    if not use_rq and not use_protect:
        rq_use = 0.0
    elif not use_protect:
        rq_use = max(rq, 0.0)
    elif not use_rq:
        rq_use = min(rq, 0.0)
    else:
        rq_use = rq
    ri_use = parts["ri"] if use_ri else 0.0
    rc_use = parts["rc"] if use_rc else 0.0
    cost = parts.get("cost", 0.0)
    return float(cfg.alpha * rq_use + cfg.beta * ri_use + cfg.gamma * rc_use - cost)
