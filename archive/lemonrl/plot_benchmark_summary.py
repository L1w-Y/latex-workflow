"""
读取 benchmark_summary.json，生成 RL vs Greedy 在测试集上的总奖励分布对比图。

输出: ../论文/figures/benchmark_distribution.png

运行（PowerShell）::

    cd D:\\PROJECTS\\latex-workflow\\lemonrl
    $env:KMP_DUPLICATE_LIB_OK="TRUE"
    & "D:\\PROJECTS\\deeplearning\\python.exe" plot_benchmark_summary.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SUMMARY_PATH = Path("../论文/figures/benchmark_summary.json")
OUT_PATH = Path("../论文/figures/benchmark_distribution.png")


def main() -> None:
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    per_tree = summary["per_tree"]
    if not per_tree:
        raise RuntimeError("per_tree is empty in benchmark_summary.json")

    rl = np.array([t["rl_total_reward"] for t in per_tree], dtype=np.float64)
    gr = np.array([t["greedy_total_reward"] for t in per_tree], dtype=np.float64)
    diff = rl - gr

    agg = summary["aggregate"]
    n = len(per_tree)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # ---------------- (a) Scatter: RL vs Greedy ----------------
    ax = axes[0]
    ax.scatter(gr, rl, s=22, alpha=0.55, color="tab:blue", edgecolor="none")
    lim_lo = float(min(rl.min(), gr.min())) - 0.25
    lim_hi = float(max(rl.max(), gr.max())) + 0.25
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color="tab:red", linestyle="--", linewidth=1.4, label="$y = x$")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Greedy total reward")
    ax.set_ylabel("Learned total reward")
    ax.set_title(f"(a) Per-tree comparison  (n = {n})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=1.0, edgecolor="0.6")
    ax.annotate(
        f"Learned win-rate = {agg['rl_win_rate'] * 100:.1f}%",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff5e6", ec="0.6"),
    )

    # ---------------- (b) Histogram of total rewards ----------------
    ax = axes[1]
    bin_lo = float(min(rl.min(), gr.min()))
    bin_hi = float(max(rl.max(), gr.max()))
    bins = np.linspace(bin_lo, bin_hi, 30)
    ax.hist(rl, bins=bins, alpha=0.6, color="tab:blue",
            label=f"Learned  ($\\mu$={agg['rl_mean_total']:+.3f}, $\\sigma$={agg['rl_std_total']:.3f})",
            edgecolor="white", linewidth=0.5)
    ax.hist(gr, bins=bins, alpha=0.6, color="tab:orange",
            label=f"Greedy  ($\\mu$={agg['greedy_mean_total']:+.3f}, $\\sigma$={agg['greedy_std_total']:.3f})",
            edgecolor="white", linewidth=0.5)
    ax.axvline(float(rl.mean()), color="tab:blue", linestyle="--", linewidth=1.1)
    ax.axvline(float(gr.mean()), color="tab:orange", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Total reward per tree")
    ax.set_ylabel("Count of test trees")
    ax.set_title("(b) Distribution of total rewards")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=9, framealpha=1.0, edgecolor="0.6")

    # ---------------- (c) Histogram of (RL - Greedy) ----------------
    ax = axes[2]
    ax.hist(diff, bins=25, color="tab:purple", alpha=0.75,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0.0, color="black", linestyle="-", linewidth=1.3, label="No difference (=0)")
    ax.axvline(float(diff.mean()), color="tab:red", linestyle="--", linewidth=1.4,
               label=f"Mean gap = {diff.mean():+.3f}")
    ax.set_xlabel(r"Learned $-$ Greedy  (total reward)")
    ax.set_ylabel("Count of test trees")
    ax.set_title("(c) Per-tree reward gap")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=9, framealpha=1.0, edgecolor="0.6")

    sampling = summary.get("sampling_method", "deterministic")
    fig.suptitle(
        f"Benchmark on {n} test trees  "
        f"(15% of {summary['num_trees_total']}-tree pool, {sampling} sampling)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {OUT_PATH.resolve()}")
    print(
        f"Summary: n={n}  |  RL mean={agg['rl_mean_total']:+.3f}  "
        f"Greedy mean={agg['greedy_mean_total']:+.3f}  "
        f"gap mean={agg['rl_minus_greedy_mean']:+.3f}  "
        f"win-rate={agg['rl_win_rate'] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
