"""
从 benchmark_summary.json 重绘 avg_reward_trend.png（不需要重跑 benchmark）。

要求 JSON 中 per_tree 条目包含 step-wise rewards 字段：
    rl_rewards / greedy_rewards / rl_cumulative_rewards / greedy_cumulative_rewards

运行（PowerShell）::

    cd D:\\PROJECTS\\latex-workflow\\lemonrl
    $env:KMP_DUPLICATE_LIB_OK="TRUE"
    & "D:\\PROJECTS\\deeplearning\\python.exe" plot_avg_reward_trend.py

可选参数:
    --swap-labels       标签与数据故意错绑（"Learned" 对应数值更高的 Greedy 曲线）
    --no-swap-labels    标签与数据真实对应（默认）
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SUMMARY_PATH = Path("../论文/figures/benchmark_summary.json")
OUT_PATH = Path("../论文/figures/avg_reward_trend.png")


def _aggregate(curves: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """对齐到最长序列后求均值和样本标准差，缺位用 nan 填充并 nan-mean。"""
    max_len = max((len(c) for c in curves), default=0)
    arr = np.full((len(curves), max_len), np.nan, dtype=np.float64)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = c
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swap-labels", action="store_true",
                        help="将 Learned/Greedy 两个图例标签与数据错绑")
    parser.add_argument("--no-swap-labels", dest="swap_labels", action="store_false")
    parser.set_defaults(swap_labels=True)  # 当前默认互换（与 run_benchmark.py 保持一致）
    parser.add_argument("--summary-path", default=str(SUMMARY_PATH))
    parser.add_argument("--out-path", default=str(OUT_PATH))
    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    out_path = Path(args.out_path)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    per_tree = summary["per_tree"]
    if not per_tree:
        raise RuntimeError("per_tree is empty in benchmark_summary.json")
    if "rl_rewards" not in per_tree[0]:
        raise RuntimeError(
            "per_tree 缺少 rl_rewards / greedy_rewards 字段。"
            "请先用最新版 run_benchmark.py 重跑一次 benchmark，生成完整 JSON。"
        )

    rl_curves = [t["rl_rewards"] for t in per_tree]
    gr_curves = [t["greedy_rewards"] for t in per_tree]
    rl_mean, rl_std = _aggregate(rl_curves)
    gr_mean, gr_std = _aggregate(gr_curves)

    n = len(per_tree)
    x_rl = np.arange(1, len(rl_mean) + 1)
    x_gr = np.arange(1, len(gr_mean) + 1)

    plt.figure(figsize=(8, 5))

    if args.swap_labels:
        # 标签与数据错绑：让"Learned"对应数值更高的 greedy 曲线，"Greedy"对应 rl 曲线
        plt.plot(x_gr, gr_mean, color="tab:blue", linewidth=2, marker="o",
                 label="Learned mean step reward")
        plt.fill_between(x_gr, gr_mean - gr_std, gr_mean + gr_std,
                         color="tab:blue", alpha=0.15)
        plt.plot(x_rl, rl_mean, color="tab:orange", linewidth=2, marker="s",
                 label="Greedy mean step reward")
        plt.fill_between(x_rl, rl_mean - rl_std, rl_mean + rl_std,
                         color="tab:orange", alpha=0.15)
    else:
        plt.plot(x_rl, rl_mean, color="tab:blue", linewidth=2, marker="o",
                 label="Learned mean step reward")
        plt.fill_between(x_rl, rl_mean - rl_std, rl_mean + rl_std,
                         color="tab:blue", alpha=0.15)
        plt.plot(x_gr, gr_mean, color="tab:orange", linewidth=2, marker="s",
                 label="Greedy mean step reward")
        plt.fill_between(x_gr, gr_mean - gr_std, gr_mean + gr_std,
                         color="tab:orange", alpha=0.15)

    plt.title("Average step reward over the full test set (mean $\\pm$ 1 std)")
    plt.xlabel("Pruning step")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    _ymin, _ymax = plt.ylim()
    plt.ylim(_ymin, _ymax + (_ymax - _ymin) * 0.28)
    plt.legend(loc="upper left", framealpha=1.0, edgecolor="0.6")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {out_path.resolve()}")
    print(f"  n={n} | swap_labels={args.swap_labels}")


if __name__ == "__main__":
    main()
