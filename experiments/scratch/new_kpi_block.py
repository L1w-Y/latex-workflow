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
