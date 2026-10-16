"""E2 师生对比：DT / BC 与专家规则的剪枝决策差异分析。

读取 evaluate.py 落盘的 eval_<tag>.csv 与 actions_<tag>.json，输出：
- 每个策略的单刀画像：刀数、单刀奖励及 R_q/R_i/R_c、命中不良枝比例、
  误剪受保护枝比例、各类别占比、被剪枝长度/半径/点数
- 与参考策略（默认 expert）逐树对比：剪除枝条集合的重合度（按枝数、按骨架点数加权）、
  首次分歧步；只被一方剪掉的枝条在初始树上的长度、半径、层级与类别构成

用法（在 dt_pruning 目录下）：
    python ../experiments/scripts/teacher_student.py --split val --exp e2_teacher --policies dt,bc,expert
结果写入同目录 teacher_student.json。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))

from config import CFG  # noqa: E402
from evaluate import result_dir  # noqa: E402
from reward import branch_context, branch_tags, compute_branch_props, detect_crossing  # noqa: E402
from tree_data import BRANCH, PARENT, build_parent_map, load_skeleton, skeleton_path  # noqa: E402

TAGS = ["cross_inferior", "cross_other", "inward", "vigorous", "drooping", "weak",
        "protect_medium_short", "protect_sturdy", "main_branch"]


def load_policy(d: str, tag: str) -> tuple[dict, dict]:
    with open(os.path.join(d, f"eval_{tag}.csv"), encoding="utf-8") as f:
        rows = {r["tree"]: r for r in csv.DictReader(f)}
    with open(os.path.join(d, f"actions_{tag}.json"), encoding="utf-8") as f:
        traces = json.load(f)
    return rows, traces


def q(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "q25": float(np.quantile(a, 0.25)), "q75": float(np.quantile(a, 0.75))}


def cut_profile(rows: dict, traces: dict) -> dict:
    cuts = [c for t in traces.values() for c in t]
    n = max(len(cuts), 1)
    tag_count = Counter(tag for c in cuts for tag in c["tags"])
    return {
        "n_trees": len(traces),
        "total_reward": q([float(r["total_reward"]) for r in rows.values()]),
        "final_pruning_ratio": q([float(r["final_pruning_ratio"]) for r in rows.values()]),
        "cuts_per_tree": q([len(t) for t in traces.values()]),
        "reward_per_cut": q([c["reward"] for c in cuts]),
        "rq_per_cut": q([c["rq"] for c in cuts]),
        "ri_per_cut": q([c["ri"] for c in cuts]),
        "rc_per_cut": q([c["rc"] for c in cuts]),
        "bad_branch_hit_rate": sum(c["rq"] > 0 for c in cuts) / n,
        "protected_cut_rate": sum(any(t.startswith("protect") for t in c["tags"]) for c in cuts) / n,
        "untagged_cut_rate": sum(not c["tags"] for c in cuts) / n,
        "tag_rate": {t: tag_count[t] / n for t in TAGS},
        "cut_length": q([c["length"] for c in cuts if c["length"] is not None]),
        "cut_radius": q([c["radius"] for c in cuts if c["radius"] is not None]),
        "points_removed_per_cut": q([c["points_removed"] for c in cuts]),
    }


class InitialTree:
    """初始树上的枝条静态信息：骨架点数、长度、半径、层级、类别标签。"""

    def __init__(self, tree: str):
        sk = load_skeleton(skeleton_path(os.path.join(CFG.data.data_root, tree)))
        bids = sk[:, BRANCH].astype(np.int64)
        self.points = dict(zip(*np.unique(bids, return_counts=True)))
        self.level = {int(b): int(sk[bids == b][0, PARENT]) for b in self.points}
        props = compute_branch_props(sk, build_parent_map(sk))
        crossing = detect_crossing(props, CFG.reward)
        ctx = branch_context(sk, props, CFG.reward)
        self.props = props
        self.tags = {b: branch_tags(b, props, crossing, *ctx, CFG.reward) for b in props}

    def describe(self, bids: set[int]) -> dict:
        bids = [b for b in bids if b in self.props]
        if not bids:
            return {"n": 0}
        n = len(bids)
        return {
            "n": n,
            "points": int(sum(self.points[b] for b in bids)),
            "length": [self.props[b].length for b in bids],
            "radius": [self.props[b].avg_radius for b in bids],
            "level": [self.level[b] for b in bids],
            "tags": Counter(t for b in bids for t in self.tags[b]),
            "untagged": sum(not self.tags[b] for b in bids),
        }


def pair_compare(a: dict, b: dict, trees: list[str], cache: dict) -> dict:
    jac, wjac, first_div, only_a, only_b = [], [], [], [], []
    for tree in trees:
        ta, tb = a[tree], b[tree]
        sa = {x for c in ta for x in c["removed"]}
        sb = {x for c in tb for x in c["removed"]}
        if tree not in cache:
            cache[tree] = InitialTree(tree)
        it = cache[tree]
        w = lambda s: sum(it.points.get(x, 0) for x in s)
        union = sa | sb
        jac.append(len(sa & sb) / max(len(union), 1))
        wjac.append(w(sa & sb) / max(w(union), 1))
        k = next((i for i, (ca, cb) in enumerate(zip(ta, tb)) if ca["bid"] != cb["bid"]),
                 min(len(ta), len(tb)))
        first_div.append(k)
        only_a.append(it.describe(sa - sb))
        only_b.append(it.describe(sb - sa))
    return {
        "n_trees": len(trees),
        "jaccard_by_branch": q(jac),
        "jaccard_by_points": q(wjac),
        "first_divergence_step": q(first_div),
        "only_ref_side": summarize_side(only_b),
        "only_policy_side": summarize_side(only_a),
    }


def summarize_side(items: list[dict]) -> dict:
    n = sum(d["n"] for d in items)
    if n == 0:
        return {"branches_per_tree": 0.0}
    cat = lambda k: [v for d in items if d["n"] for v in d[k]]
    tags = sum((d["tags"] for d in items if d["n"]), Counter())
    return {
        "branches_per_tree": n / len(items),
        "points_per_tree": sum(d.get("points", 0) for d in items) / len(items),
        "length": q(cat("length")),
        "radius": q(cat("radius")),
        "level": q(cat("level")),
        "tag_rate": {t: tags[t] / n for t in TAGS},
        "untagged_rate": sum(d.get("untagged", 0) for d in items) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--stage", choices=["dev", "paper"], default="dev")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--policies", default="dt,bc,expert", help="eval_<tag> 的 tag，逗号分隔")
    ap.add_argument("--ref", default="expert")
    args = ap.parse_args()

    d = result_dir(args.stage, args.exp, args.split)
    tags = [t for t in args.policies.split(",") if t]
    data = {t: load_policy(d, t) for t in tags}
    common = sorted(set.intersection(*(set(tr) for _, tr in data.values())))
    print(f"{d}: {len(common)} 棵共同树")

    report = {"trees": len(common), "profiles": {}, "vs_" + args.ref: {}}
    for t, (rows, traces) in data.items():
        report["profiles"][t] = cut_profile({k: rows[k] for k in common},
                                            {k: traces[k] for k in common})
    cache: dict = {}
    ref_traces = data[args.ref][1]
    for t, (_, traces) in data.items():
        if t != args.ref:
            report["vs_" + args.ref][t] = pair_compare(traces, ref_traces, common, cache)

    out = os.path.join(d, "teacher_student.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("已保存:", out)


if __name__ == "__main__":
    main()
