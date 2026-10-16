"""E2 师生对比两张图。

图 1 e2_fig_return_curve：累计回报—剪枝进度曲线。横轴为累计剪枝比，纵轴为每根初始分枝的累计回报
      （按树规模归一，消除大树回报天然更高的影响）；某策略停手后保持其最终值，
      即"剪枝比预算为 x 时该策略取得的回报"。均值 ± 95% CI，圆点标出各策略平均停止位置。
图 2 e2_fig_removed_diff：剪除枝条差异。(a) 两者都剪 / 仅 DT 剪 / 仅专家剪三类枝条在初始树上的类别构成；
      (b) 单树案例，按三类给被剪枝条上色。

读取 results/<stage>/<exp>/<split>/ 下的 actions_<tag>.json 与 eval_<tag>.csv，输出 PDF + PNG 到同目录。

用法（在 dt_pruning 目录下）：
    python ../experiments/scripts/e2_figures.py --split val --exp e2_teacher
    python ../experiments/scripts/e2_figures.py --split val --exp e2_teacher --case Lemon00_21
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "dt_pruning")))
sys.path.insert(0, _HERE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from config import CFG  # noqa: E402
from evaluate import result_dir  # noqa: E402
from teacher_student import InitialTree  # noqa: E402
from tree_data import BRANCH, load_skeleton, skeleton_path  # noqa: E402

COLORS = {"dt": "#0072B2", "expert": "#D55E00", "bc": "#009E73"}
LABELS = {"dt": "DT", "expert": "Expert rule (teacher)", "bc": "BC"}
SET_COLORS = {"both": "#4D4D4D", "dt_only": COLORS["dt"], "expert_only": COLORS["expert"],
              "kept": "#C8C8C8"}
SET_LABELS = {"both": "Removed by both", "dt_only": "DT only", "expert_only": "Expert only"}
CATEGORIES = [("cross_inferior", "Crossing (inferior)"), ("cross_other", "Crossing (other)"),
              ("inward", "Inward"), ("drooping", "Drooping"), ("weak", "Weak"),
              ("protected", "Protected functional"), ("main_branch", "Main scaffold")]

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 9,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "savefig.dpi": 300,
})


def load(d: str, tag: str):
    with open(os.path.join(d, f"actions_{tag}.json"), encoding="utf-8") as f:
        traces = json.load(f)
    with open(os.path.join(d, f"eval_{tag}.csv"), encoding="utf-8") as f:
        rows = {r["tree"]: r for r in csv.DictReader(f)}
    return traces, rows


def n_branches(tree: str) -> int:
    sk = load_skeleton(skeleton_path(os.path.join(CFG.data.data_root, tree)))
    return len(np.unique(sk[:, BRANCH]))


def save(fig, d: str, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已保存:", os.path.join(d, name + ".pdf/.png"))


# ------------------------------------------------------------------------------
# 图 1
# ------------------------------------------------------------------------------

def return_curve(d: str, data: dict, trees: list[str], n0: dict):
    grid = np.linspace(0, CFG.traj.max_pruning_ratio + 0.02, 161)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for tag in ("dt", "expert", "bc"):
        traces, rows = data[tag]
        curves, ends = [], []
        for t in trees:
            ratio = np.cumsum([len(c["removed"]) for c in traces[t]]) / n0[t]
            ret = np.cumsum([c["reward"] for c in traces[t]]) / n0[t]
            idx = np.searchsorted(ratio, grid, side="right") - 1
            curves.append(np.where(idx >= 0, ret[np.clip(idx, 0, None)], 0.0))
            ends.append((ratio[-1] if len(ratio) else 0.0, ret[-1] if len(ret) else 0.0))
        c = np.array(curves)
        m, se = c.mean(0), c.std(0, ddof=1) / np.sqrt(len(c))
        ax.plot(grid, m, color=COLORS[tag], lw=1.4, label=LABELS[tag])
        ax.fill_between(grid, m - 1.96 * se, m + 1.96 * se, color=COLORS[tag], alpha=0.18, lw=0)
        ex = np.mean([e[0] for e in ends])
        ax.plot(ex, np.interp(ex, grid, m), "o", ms=4, color=COLORS[tag],
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    ax.axvline(CFG.traj.max_pruning_ratio, color="0.5", lw=0.6, ls="--")
    ax.text(CFG.traj.max_pruning_ratio, ax.get_ylim()[1] * 0.04, " ratio cap", color="0.4",
            fontsize=7, ha="left", va="bottom", rotation=90)
    ax.set_xlabel("Cumulative pruning ratio")
    ax.set_ylabel("Cumulative return per initial branch")
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, d, "e2_fig_return_curve")


# ------------------------------------------------------------------------------
# 图 2
# ------------------------------------------------------------------------------

def removed_sets(traces: dict, tree: str) -> set[int]:
    return {x for c in traces[tree] for x in c["removed"]}


def tag_of(it: InitialTree, b: int, key: str) -> bool:
    tags = it.tags.get(b, [])
    if key == "protected":
        return any(t.startswith("protect") for t in tags)
    return key in tags


def category_shares(d_traces, e_traces, trees, cache):
    groups = {k: [] for k in SET_LABELS}
    for t in trees:
        it = cache.setdefault(t, InitialTree(t))
        a, b = removed_sets(d_traces, t), removed_sets(e_traces, t)
        for k, s in (("both", a & b), ("dt_only", a - b), ("expert_only", b - a)):
            groups[k] += [(it, x) for x in s if x in it.props]
    shares = {k: [np.mean([tag_of(it, x, c) for it, x in v]) if v else 0.0 for c, _ in CATEGORIES]
              for k, v in groups.items()}
    per_tree = {k: len(v) / len(trees) for k, v in groups.items()}
    return shares, per_tree


def pick_case(data, trees) -> str:
    diff = {t: float(data["dt"][1][t]["total_reward"]) - float(data["expert"][1][t]["total_reward"])
            for t in trees}
    pos = sorted((v, t) for t, v in diff.items() if v > 0)
    return pos[len(pos) // 2][1] if pos else trees[0]


def draw_tree(ax, tree: str, a: set[int], b: set[int]):
    """侧视投影：水平面取骨架点主方向为横轴，竖直为纵轴。"""
    sk = load_skeleton(skeleton_path(os.path.join(CFG.data.data_root, tree)))
    bids = sk[:, BRANCH].astype(np.int64)
    xy = sk[:, :2] - sk[:, :2].mean(0)
    axis = np.linalg.svd(xy, full_matrices=False)[2][0]
    proj = np.column_stack([xy @ axis, sk[:, 2]])
    segs = {k: [] for k in SET_COLORS}
    for bid in np.unique(bids):
        pts = proj[bids == bid]
        if len(pts) < 2:
            continue
        key = ("both" if bid in a and bid in b else "dt_only" if bid in a
               else "expert_only" if bid in b else "kept")
        segs[key] += list(zip(pts[:-1], pts[1:]))
    for key, lw, z in (("kept", 0.5, 1), ("both", 0.9, 2), ("expert_only", 1.1, 3), ("dt_only", 1.1, 4)):
        if segs[key]:
            ax.add_collection(LineCollection(segs[key], colors=SET_COLORS[key], linewidths=lw, zorder=z))
    pad = 0.03 * np.ptp(proj, axis=0).max()
    ax.set_xlim(proj[:, 0].min() - pad, proj[:, 0].max() + pad)
    ax.set_ylim(proj[:, 1].min() - pad, proj[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()


def removed_diff(d: str, data: dict, trees: list[str], case: str | None):
    cache: dict = {}
    shares, per_tree = category_shares(data["dt"][0], data["expert"][0], trees, cache)
    case = case or pick_case(data, trees)

    fig = plt.figure(figsize=(7.2, 3.4))
    ax = fig.add_axes([0.14, 0.13, 0.42, 0.78])
    y = np.arange(len(CATEGORIES))
    h = 0.26
    for i, k in enumerate(("both", "dt_only", "expert_only")):
        ax.barh(y + (i - 1) * h, np.array(shares[k]) * 100, height=h, color=SET_COLORS[k],
                label=f"{SET_LABELS[k]} ({per_tree[k]:.1f} / tree)")
    ax.set_yticks(y, [n for _, n in CATEGORIES])
    ax.invert_yaxis()
    ax.set_xlabel("Share of removed branches with the attribute (%)")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(a) Attributes of removed branches (initial tree)", fontsize=9, loc="left")

    ax3 = fig.add_axes([0.60, 0.13, 0.39, 0.78])
    a, b = removed_sets(data["dt"][0], case), removed_sets(data["expert"][0], case)
    draw_tree(ax3, case, a, b)
    rd = float(data["dt"][1][case]["total_reward"]); re_ = float(data["expert"][1][case]["total_reward"])
    ax3.set_title(f"(b) Case tree {case} (side view)\nreturn: DT {rd:.0f} vs. expert {re_:.0f}",
                  fontsize=9, loc="left")
    handles = [plt.Line2D([], [], color=SET_COLORS[k], lw=1.6, label=l) for k, l in
               (("kept", "Retained"), ("both", "Removed by both"), ("dt_only", "DT only"),
                ("expert_only", "Expert only"))]
    ax3.legend(handles=handles, loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.4, columnspacing=1.0)
    save(fig, d, "e2_fig_removed_diff")
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--stage", choices=["dev", "paper"], default="dev")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--case", default=None, help="案例树；默认取 DT 胜出的树中回报差居中者")
    args = ap.parse_args()

    d = result_dir(args.stage, args.exp, args.split)
    data = {t: load(d, t) for t in ("dt", "expert", "bc")}
    trees = sorted(set.intersection(*(set(tr) for tr, _ in data.values())))
    n0 = {t: n_branches(t) for t in trees}
    return_curve(d, data, trees, n0)
    case = removed_diff(d, data, trees, args.case)
    print("案例树:", case)


if __name__ == "__main__":
    main()
