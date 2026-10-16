"""稳定性候选口径：骨干枝负载力矩。

每根骨干枝（初始树固定；L > 0.5 m 且 r̄ > 0.02 m，主干除外）：
    M_b = Σ_子树枝段 长度 × 枝段中点到该骨干枝基部的水平距离
    ρ_b = M_b / r_base³
S3 = mean_b exp(−ρ_b / ρ̄_0)，ρ̄_0 为该树剪前 ρ 均值；被剪掉的骨干枝不计入均值。

用法（dt_pruning 目录下）：
    python ../experiments/scratch/kpi_proto_s3.py --trees 10 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))

from config import CFG  # noqa: E402
from evaluate import _Env, result_dir  # noqa: E402
from graph_cache import load_val_split  # noqa: E402
from tree_data import trunk_branch_id  # noqa: E402

cfg = CFG.reward


def scaffold_ids(props, trunk):
    return [b for b, p in props.items() if b != trunk
            and p.length > cfg.main_branch_length and p.avg_radius > cfg.main_branch_radius]


def limb_ratios(props, scaffolds):
    children = {}
    for b, p in props.items():
        children.setdefault(p.parent_bid, []).append(b)
    out = {}
    for s in scaffolds:
        if s not in props:
            continue
        base = props[s].start[:2]
        r_base = max(props[s].avg_radius, 1e-4)
        stack, m = [s], 0.0
        while stack:
            b = stack.pop()
            pts = props[b].points
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            mid = 0.5 * (pts[:-1, :2] + pts[1:, :2])
            m += float((seg * np.linalg.norm(mid - base, axis=1)).sum())
            stack.extend(children.get(b, []))
        out[s] = m / r_base ** 3
    return out


def s3(props, scaffolds, rho0):
    r = limb_ratios(props, scaffolds)
    if not r:
        return 1.0
    return float(np.mean([np.exp(-v / rho0) for v in r.values()]))


def _replay(job):
    tree, policy, bids = job
    env = _Env(os.path.join(CFG.data.data_root, tree), CFG.eval.seed)
    sc = scaffold_ids(env.props, trunk_branch_id(env.state.skeleton))
    rho0 = float(np.mean(list(limb_ratios(env.props, sc).values())))
    k0 = s3(env.props, sc, rho0); first = k0
    cuts = []
    for b in bids:
        env.step(int(b))
        k1 = s3(env.props, sc, rho0)
        c = env.trace[-1]
        cuts.append({"tags": c["tags"] or ["none"], "dS3": k1 - k0})
        k0 = k1
    return tree, policy, first, k0, len(sc), cuts


TAGS = ["cross_inferior", "cross_other", "inward", "vigorous", "drooping", "weak",
        "protect_medium_short", "protect_sturdy", "main_branch", "none"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--policies", default="expert,dt,random")
    args = ap.parse_args()
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
    delta = {}
    for p in pols:
        rs = sorted([r for r in res if r[1] == p])
        delta[p] = np.array([r[3] - r[2] for r in rs])
        cuts = [c for r in rs for c in r[5]]
        print(p, f"剪前 {np.mean([r[2] for r in rs]):.4f} 剪后 {np.mean([r[3] for r in rs]):.4f}",
              f"改善树 {np.mean(delta[p] > 0):.0%} 骨干枝 {np.mean([r[4] for r in rs]):.1f} 根",
              f"单刀|d|中位 {np.median(np.abs([c['dS3'] for c in cuts])):.2e}")
        for t in TAGS:
            cs = [c["dS3"] for c in cuts if next((x for x in TAGS if x in c["tags"]), "none") == t]
            if cs:
                print(f"    {t:<22} {len(cs):>5} dS3 {np.mean(cs):+.5f}")
    for a, b in (("expert", "random"), ("dt", "random"), ("dt", "expert")):
        if a in delta and b in delta:
            x, y = delta[a], delta[b]
            print(f"{a}−{b} {np.mean(x - y):+.4f} 胜 {np.mean(x > y):.0%} p {wilcoxon(x, y).pvalue:.3f}")


if __name__ == "__main__":
    main()
