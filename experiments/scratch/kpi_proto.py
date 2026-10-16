"""KPI 新口径原型（只做验证，不改 reward.py）。

新口径：
  L2 内膛受光：初始树上固定采样一批枝段点作为受光点，沿 5 个方向（竖直 + 4 个 30° 天顶角）
     向上投射，被其他枝段（半径 = 枝半径 + 叶幕包络 r_leaf）挡住即不受光；L2 = 受光比例
  C2 局部拥挤：有效枝上的点，d_c 内存在其他枝条的点即记为拥挤（排除分叉处 d_c 内的近邻）；
     C2 = 1 − 拥挤点比例（越高越通透）
  P2 结果潜力均值：P_b 对有效非主干枝取平均（原为求和）

用法（dt_pruning 目录下）：
    python ../experiments/scratch/kpi_proto.py scan --trees 10
    python ../experiments/scratch/kpi_proto.py replay --trees 10 --workers 8 --r-leaf 0.2 --d-c 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))

from config import CFG  # noqa: E402
from evaluate import _Env, result_dir  # noqa: E402
from graph_cache import load_val_split  # noqa: E402
from reward import compute_branch_props, kpi_potential  # noqa: E402
from tree_data import build_parent_map, load_skeleton, skeleton_path, trunk_branch_id  # noqa: E402

N_TARGETS = 400
ZENITH = np.deg2rad(30.0)
DIRS = np.array([[0, 0, 1]] + [[np.sin(ZENITH) * np.cos(a), np.sin(ZENITH) * np.sin(a), np.cos(ZENITH)]
                                for a in np.deg2rad([0, 90, 180, 270])])


def _segments(props):
    a, b, r, bid = [], [], [], []
    for p in props.values():
        if p.points.shape[0] < 2:
            continue
        a.append(p.points[:-1]); b.append(p.points[1:])
        r.append(np.full(p.points.shape[0] - 1, p.avg_radius))
        bid.append(np.full(p.points.shape[0] - 1, p.branch_id))
    return np.concatenate(a), np.concatenate(b), np.concatenate(r), np.concatenate(bid)


def make_targets(props, seed=0):
    tb = None
    pts, bids = [], []
    for p in props.values():
        if p.length <= CFG.reward.l_eff:
            continue
        pts.append(p.points); bids.append(np.full(p.points.shape[0], p.branch_id))
    pts, bids = np.concatenate(pts), np.concatenate(bids)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=min(N_TARGETS, len(pts)), replace=False)
    return pts[idx], bids[idx]


def light2(props, targets, r_leaf):
    tp, tb = targets
    alive = np.array([b in props for b in tb])
    tp, tb = tp[alive], tb[alive]
    if len(tp) == 0:
        return 1.0
    A, B, Rr, Sb = _segments(props)
    Rr = Rr + r_leaf
    lit = np.zeros((len(tp), len(DIRS)), dtype=bool)
    for k, d in enumerate(DIRS):
        u = np.cross(d, [1.0, 0, 0]) if abs(d[0]) < 0.9 else np.cross(d, [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        P = np.stack([A @ u, A @ v], 1); Q = np.stack([B @ u, B @ v], 1)
        ta, tbb = A @ d, B @ d
        T = np.stack([tp @ u, tp @ v], 1); tt = tp @ d
        seg = Q - P
        den = np.maximum((seg ** 2).sum(1), 1e-12)
        for i0 in range(0, len(T), 100):
            Ti = T[i0:i0 + 100]
            w = np.clip(((Ti[:, None, :] - P[None]) * seg[None]).sum(-1) / den[None], 0, 1)
            closest = P[None] + w[..., None] * seg[None]
            d2 = ((Ti[:, None, :] - closest) ** 2).sum(-1)
            tc = ta[None] + w * (tbb - ta)[None]
            hit = (d2 <= Rr[None] ** 2) & (tc > tt[i0:i0 + 100, None] + 1e-6) \
                & (Sb[None] != tb[i0:i0 + 100, None])
            lit[i0:i0 + 100, k] = ~hit.any(1)
    return float(lit.mean())


def crowd2(props, d_c):
    pts, bids, starts = [], [], {}
    for p in props.values():
        if p.length <= CFG.reward.l_eff:
            continue
        pts.append(p.points); bids.append(np.full(p.points.shape[0], p.branch_id))
        starts[p.branch_id] = p.start
    if not pts:
        return 1.0
    pts, bids = np.concatenate(pts), np.concatenate(bids)
    st = np.array([starts[b] for b in bids])
    tree = cKDTree(pts)
    pairs = tree.query_pairs(d_c, output_type="ndarray")
    if len(pairs) == 0:
        return 1.0
    i, j = pairs[:, 0], pairs[:, 1]
    other = bids[i] != bids[j]
    near_fork = (np.linalg.norm(pts[i] - st[j], axis=1) < d_c) | (np.linalg.norm(pts[j] - st[i], axis=1) < d_c)
    ok = other & ~near_fork
    crowded = np.zeros(len(pts), dtype=bool)
    crowded[i[ok]] = True; crowded[j[ok]] = True
    return float(1.0 - crowded.mean())


def potential2(props, trunk_bid):
    eff = [p for p in props.values() if p.branch_id != trunk_bid and p.length >= CFG.reward.l_eff]
    if not eff:
        return 0.0
    return kpi_potential(props, CFG.reward, trunk_bid) / len(eff)


def scan(args):
    names = load_val_split()["val"][:args.trees]
    out = {}
    for n in names:
        sk = load_skeleton(skeleton_path(os.path.join(CFG.data.data_root, n)))
        props = compute_branch_props(sk, build_parent_map(sk))
        tg = make_targets(props)
        row = {f"L2_r{r}": light2(props, tg, r) for r in (0.0, 0.05, 0.1, 0.2, 0.3)}
        row.update({f"C2_d{d}": crowd2(props, d) for d in (0.1, 0.2, 0.3, 0.5)})
        row["P2"] = potential2(props, trunk_branch_id(sk))
        out[n] = row
        print(n, {k: round(v, 3) for k, v in row.items()}, flush=True)
    keys = list(next(iter(out.values())))
    print("mean", {k: round(float(np.mean([o[k] for o in out.values()])), 3) for k in keys})


_W = {}


def _init(r_leaf, d_c):
    _W.update(r_leaf=r_leaf, d_c=d_c)


def _replay(job):
    tree, policy, bids = job
    env = _Env(os.path.join(CFG.data.data_root, tree), CFG.eval.seed)
    tg = make_targets(env.props)
    tb = trunk_branch_id(env.state.skeleton)

    def kp():
        return {"L2": light2(env.props, tg, _W["r_leaf"]), "C2": crowd2(env.props, _W["d_c"]),
                "P2": potential2(env.props, tb)}
    k0 = kp(); first = dict(k0)
    cuts = []
    for b in bids:
        env.step(int(b))
        k1 = kp()
        c = env.trace[-1]
        cuts.append({"tags": c["tags"] or ["none"], "points": c["points_removed"],
                     **{f"d{k}": k1[k] - k0[k] for k in k1}})
        k0 = k1
    return tree, policy, first, k0, cuts


TAGS = ["cross_inferior", "cross_other", "inward", "vigorous", "drooping", "weak",
        "protect_medium_short", "protect_sturdy", "main_branch", "none"]


def replay(args):
    d = result_dir("dev", "e3_baselines", "val")
    names = load_val_split()["val"][:args.trees]
    jobs = []
    for p in args.policies.split(","):
        acts = json.load(open(os.path.join(d, f"actions_{p}.json"), encoding="utf-8"))
        jobs += [(n, p, [c["bid"] for c in acts[n]]) for n in names]
    t0 = time.time()
    res = []
    with Pool(args.workers, initializer=_init, initargs=(args.r_leaf, args.d_c)) as pool:
        for i, r in enumerate(pool.imap_unordered(_replay, jobs), 1):
            res.append(r)
            print(f"[{i}/{len(jobs)}] {r[1]} {r[0]} {(time.time() - t0) / 60:.1f} min", flush=True)
    summary = {"r_leaf": args.r_leaf, "d_c": args.d_c, "trees": names, "policy": {}}
    for p in args.policies.split(","):
        rs = [r for r in res if r[1] == p]
        cuts = [c for r in rs for c in r[4]]
        s = {"before": {k: float(np.mean([r[2][k] for r in rs])) for k in ("L2", "C2", "P2")},
             "after": {k: float(np.mean([r[3][k] for r in rs])) for k in ("L2", "C2", "P2")},
             "per_tree_delta": {k: [r[3][k] - r[2][k] for r in sorted(rs)] for k in ("L2", "C2", "P2")},
             "cut_abs_median": {k: float(np.median(np.abs([c[f"d{k}"] for c in cuts]))) for k in ("L2", "C2", "P2")},
             "by_tag": {}}
        for t in TAGS:
            cs = [c for c in cuts if next((x for x in TAGS if x in c["tags"]), "none") == t]
            if cs:
                s["by_tag"][t] = {"n": len(cs), **{f"d{k}": float(np.mean([c[f"d{k}"] for c in cs]))
                                                    for k in ("L2", "C2", "P2")}}
        summary["policy"][p] = s
    path = os.path.join(os.path.dirname(__file__), f"kpi_proto_r{args.r_leaf}_d{args.d_c}.json")
    json.dump(summary, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for p, s in summary["policy"].items():
        print(p, "before", {k: round(v, 4) for k, v in s["before"].items()},
              "after", {k: round(v, 4) for k, v in s["after"].items()},
              "cut|d| med", {k: f"{v:.2e}" for k, v in s["cut_abs_median"].items()})
        for t, b in s["by_tag"].items():
            print("   ", t.ljust(22), b["n"], {k: f"{v:+.4f}" for k, v in b.items() if k != "n"})
    print("写入", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "replay"])
    ap.add_argument("--trees", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--r-leaf", type=float, default=0.2)
    ap.add_argument("--d-c", type=float, default=0.3)
    ap.add_argument("--policies", default="expert,dt,random")
    args = ap.parse_args()
    scan(args) if args.mode == "scan" else replay(args)


if __name__ == "__main__":
    main()
