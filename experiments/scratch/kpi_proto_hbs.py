"""层次性 H / 平衡 B / 稳定性 S 的机制诊断与候选口径。

scan   ：初始树上 r² 权重在主干、骨干枝、其余枝的占比
replay ：重放 expert / dt / random 动作，记录
         - 旧 H 的两部分 H_global、H_main 的单刀变化，以及被剪枝在全树中的高度、半径分位
         - 旧 B、S（r² 加权）与候选 B2、S2（枝长加权，代表枝叶与挂果负载；偏移按冠幅归一）
         - 候选 H2：冠层上下分层的负载比例（下层负载占比，越高越"下大上小"）

用法（dt_pruning 目录下）：
    python ../experiments/scratch/kpi_proto_hbs.py scan --trees 6
    python ../experiments/scratch/kpi_proto_hbs.py replay --trees 10 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))

from config import CFG  # noqa: E402
from evaluate import _Env, result_dir  # noqa: E402
from graph_cache import load_val_split  # noqa: E402
from reward import compute_branch_props, kpi_balance, kpi_stability  # noqa: E402
from tree_data import R, Z, build_parent_map, load_skeleton, skeleton_path, trunk_branch_id  # noqa: E402

cfg = CFG.reward


def h_parts(props):
    eff = [p for p in props.values() if p.length > 0.01]
    rho, _ = spearmanr([p.avg_z for p in eff], [p.avg_radius for p in eff])
    hg = 0.5 if np.isnan(rho) else (1.0 - rho) / 2.0
    mains = sorted([p for p in props.values()
                    if p.length > cfg.main_branch_length and p.avg_radius > cfg.main_branch_radius],
                   key=lambda p: p.avg_z)
    if len(mains) < 2:
        hm = 0.5
    else:
        m = len(mains)
        hl = sum(mains[i].length >= mains[i + 1].length for i in range(m - 1)) / (m - 1)
        hr = sum(mains[i].avg_radius >= mains[i + 1].avg_radius for i in range(m - 1)) / (m - 1)
        hm = (hl + hr) / 2.0
    return hg, hm, len(mains)


class Frame:
    """初始树的参照：主干基部、冠层高度区间、冠幅（固定，各状态复用）。"""

    def __init__(self, sk):
        self.base = sk[np.argmin(sk[:, Z])][:3]
        tb = trunk_branch_id(sk)
        from tree_data import BRANCH
        non_trunk = sk[sk[:, BRANCH] != tb]
        self.z_lo = float(np.percentile(non_trunk[:, Z], 2))
        self.z_hi = float(non_trunk[:, Z].max())
        self.z_mid = 0.5 * (self.z_lo + self.z_hi)
        self.spread = float(np.percentile(np.linalg.norm(non_trunk[:, :2] - self.base[:2], axis=1), 90))
        self.trunk = tb


def load_points(props, frame):
    """负载代理：非主干有效枝的枝段中点，权重 = 枝段长度。"""
    mids, w = [], []
    for p in props.values():
        if p.branch_id == frame.trunk or p.length <= cfg.l_eff:
            continue
        seg = np.diff(p.points, axis=0)
        mids.append(0.5 * (p.points[:-1] + p.points[1:]))
        w.append(np.linalg.norm(seg, axis=1))
    if not mids:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(mids), np.concatenate(w)


def b2(props, frame, k=2.0):
    m, w = load_points(props, frame)
    if w.sum() <= 0:
        return 0.5
    g = (m[:, :2] * w[:, None]).sum(0) / w.sum()
    return float(np.exp(-k * np.linalg.norm(g - frame.base[:2]) / frame.spread))


def s2(props, frame, k=2.0):
    m, w = load_points(props, frame)
    if w.sum() <= 0:
        return 0.5
    g = (m * w[:, None]).sum(0) / w.sum()
    h = max(frame.z_hi - frame.base[2], 1e-6)
    s_z = 1.0 - (g[2] - frame.base[2]) / h
    s_xy = np.exp(-k * np.linalg.norm(g[:2] - frame.base[:2]) / frame.spread)
    return float(cfg.w_z * s_z + cfg.w_xy * s_xy)


def h2(props, frame):
    m, w = load_points(props, frame)
    if w.sum() <= 0:
        return 0.5
    return float(w[m[:, 2] < frame.z_mid].sum() / w.sum())


def scan(args):
    for n in load_val_split()["val"][:args.trees]:
        sk = load_skeleton(skeleton_path(os.path.join(CFG.data.data_root, n)))
        props = compute_branch_props(sk, build_parent_map(sk))
        tb = trunk_branch_id(sk)
        w = {"trunk": 0.0, "scaffold": 0.0, "other": 0.0}
        for p in props.values():
            mass = float((p.avg_radius ** 2) * p.points.shape[0])
            if p.branch_id == tb:
                w["trunk"] += mass
            elif p.length > cfg.main_branch_length and p.avg_radius > cfg.main_branch_radius:
                w["scaffold"] += mass
            else:
                w["other"] += mass
        tot = sum(w.values())
        hg, hm, nm = h_parts(props)
        fr = Frame(sk)
        print(n, "r² 权重占比", {k: f"{v / tot:.1%}" for k, v in w.items()},
              f"| 骨干枝 {nm} 根 / 共 {len(props)} | H_g {hg:.3f} H_m {hm:.3f}",
              f"| B {kpi_balance(sk, cfg):.3f} B2 {b2(props, fr):.3f}",
              f"| S {kpi_stability(sk, cfg):.3f} S2 {s2(props, fr):.3f} | H2 {h2(props, fr):.3f}", flush=True)


def _replay(job):
    tree, policy, bids = job
    env = _Env(os.path.join(CFG.data.data_root, tree), CFG.eval.seed)
    fr = Frame(env.state.skeleton)

    def kp():
        hg, hm, nm = h_parts(env.props)
        return {"Hg": hg, "Hm": hm, "n_main": nm, "B": kpi_balance(env.state.skeleton, cfg),
                "S": kpi_stability(env.state.skeleton, cfg), "B2": b2(env.props, fr),
                "S2": s2(env.props, fr), "H2": h2(env.props, fr)}
    k0 = kp(); first = dict(k0)
    cuts = []
    for b in bids:
        props = env.props
        eff = [p for p in props.values() if p.length > 0.01]
        zs = np.array([p.avg_z for p in eff]); rs = np.array([p.avg_radius for p in eff])
        p = props[int(b)]
        zq = float((zs < p.avg_z).mean()); rq = float((rs < p.avg_radius).mean())
        env.step(int(b))
        k1 = kp()
        c = env.trace[-1]
        cuts.append({"tags": c["tags"] or ["none"], "zq": zq, "rq": rq,
                     **{f"d{k}": k1[k] - k0[k] for k in k1}})
        k0 = k1
    return tree, policy, first, k0, cuts


KEYS = ("Hg", "Hm", "B", "S", "B2", "S2", "H2")


def replay(args):
    d = result_dir("dev", "e3_baselines", "val")
    names = load_val_split()["val"][:args.trees]
    pols = args.policies.split(",")
    jobs = []
    for p in pols:
        acts = json.load(open(os.path.join(d, f"actions_{p}.json"), encoding="utf-8"))
        jobs += [(n, p, [c["bid"] for c in acts[n]]) for n in names]
    t0 = time.time(); res = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_replay, jobs), 1):
            res.append(r)
            print(f"[{i}/{len(jobs)}] {r[1]} {r[0]} {(time.time() - t0) / 60:.1f} min", flush=True)
    out = {}
    for p in pols:
        rs = sorted([r for r in res if r[1] == p])
        cuts = [c for r in rs for c in r[4]]
        dl = {k: np.array([r[3][k] - r[2][k] for r in rs]) for k in KEYS}
        zq = np.array([c["zq"] for c in cuts]); rq = np.array([c["rq"] for c in cuts])
        dhg = np.array([c["dHg"] for c in cuts])
        out[p] = {
            "before": {k: float(np.mean([r[2][k] for r in rs])) for k in KEYS + ("n_main",)},
            "after": {k: float(np.mean([r[3][k] for r in rs])) for k in KEYS + ("n_main",)},
            "delta_trees": {k: v.tolist() for k, v in dl.items()},
            "cut_zq_mean": float(zq.mean()), "cut_rq_mean": float(rq.mean()),
            "corr_dHg_zq": float(spearmanr(zq, dhg)[0]), "corr_dHg_rq": float(spearmanr(rq, dhg)[0]),
            "dHm_nonzero": float(np.mean([c["dHm"] != 0 for c in cuts])),
            "dHg_low_thin": float(dhg[(zq < 0.5) & (rq < 0.5)].mean()) if ((zq < 0.5) & (rq < 0.5)).any() else None,
            "dHg_high_thin": float(dhg[(zq >= 0.5) & (rq < 0.5)].mean()) if ((zq >= 0.5) & (rq < 0.5)).any() else None,
            "n_cuts": len(cuts),
        }
        o = out[p]
        print(p, "| 剪前→剪后", {k: f"{o['before'][k]:.4f}→{o['after'][k]:.4f}" for k in KEYS + ("n_main",)})
        print("   被剪枝 高度分位 %.2f 半径分位 %.2f | dHg~高度 %.2f dHg~半径 %.2f | 低细枝 dHg %s 高细枝 dHg %s | H_main 变动刀占比 %.3f"
              % (o["cut_zq_mean"], o["cut_rq_mean"], o["corr_dHg_zq"], o["corr_dHg_rq"],
                 f"{o['dHg_low_thin']:+.5f}" if o["dHg_low_thin"] is not None else "-",
                 f"{o['dHg_high_thin']:+.5f}" if o["dHg_high_thin"] is not None else "-", o["dHm_nonzero"]))
    if "expert" in out and "random" in out:
        for k in KEYS:
            e, r = np.array(out["expert"]["delta_trees"][k]), np.array(out["random"]["delta_trees"][k])
            dd = np.array(out["dt"]["delta_trees"][k]) if "dt" in out else None
            msg = f"{k}: 专家−随机 {np.mean(e - r):+.4f} 胜 {np.mean(e > r):.0%} p {wilcoxon(e, r).pvalue if np.any(e != r) else float('nan'):.3f}"
            if dd is not None:
                msg += f" | DT−随机 {np.mean(dd - r):+.4f} 胜 {np.mean(dd > r):.0%} p {wilcoxon(dd, r).pvalue if np.any(dd != r) else float('nan'):.3f}"
            print(msg)
    path = os.path.join(os.path.dirname(__file__), "kpi_proto_hbs.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("写入", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "replay"])
    ap.add_argument("--trees", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--policies", default="expert,dt,random")
    args = ap.parse_args()
    scan(args) if args.mode == "scan" else replay(args)


if __name__ == "__main__":
    main()
