"""评价维度审计：全部用已有数据，不新增评估。
A 训练数据的停止标签 vs DT 实际停止位置
B 同剪枝量比较：DT 终点 KPI vs 其他策略在 DT 停止剪枝比处的 KPI（曲线插值）
C 各指标区分力：理想区间命中、<0.25、<0.18
E DT 停得早的树有什么特征（树规模、目标回报）
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "dt_pruning")))
from config import CFG  # noqa: E402

R = os.path.join(HERE, "..", "results", "dev")
KEYS = ["hierarchy", "crowding", "balance", "light", "potential", "stability"]
cfg = CFG.reward


def ksum(row, suffix="_delta"):
    return sum(cfg.kpi_weights[k] * float(row[k + suffix]) / cfg.kpi_scales[k] for k in KEYS)


def load(exp, tag):
    with open(os.path.join(R, exp, "val", f"eval_{tag}.csv"), encoding="utf-8") as f:
        return {r["tree"]: r for r in csv.DictReader(f)}


# ---------------- A
tiers = np.array(CFG.traj.expert_target_tiers)
print("A. 训练数据专家停止标签：目标档", tiers.tolist(), f"均值 {tiers.mean():.3f}（按树序号轮转，与树状态无关）")
try:
    from graph_cache import load_cache, load_val_split
    tr = load_val_split()["train"][:60]
    ex = [t for t in load_cache(tr, ("expert",), reward_set="full")]
    fr = np.array([t["final_ratio"] for t in ex if t.get("final_ratio") is not None])
    tg = np.array([t["target_ratio"] for t in ex if t.get("target_ratio") is not None])
    print(f"   训练专家轨迹（{len(fr)} 条）实际终点剪枝比均值 {fr.mean():.3f}，目标档均值 {tg.mean():.3f}")
except Exception as e:  # noqa: BLE001
    print("   读取训练轨迹失败：", repr(e)[:120])
base = load("stop_diag", "dt_base")
m06 = load("stop_diag", "dt_bias_m06")
trees = sorted(base)
rb = np.array([float(base[n]["final_pruning_ratio"]) for n in trees])
rm = np.array([float(m06[n]["final_pruning_ratio"]) for n in trees])
print(f"   DT 实际终点剪枝比：未校准 {rb.mean():.3f}，bias −0.6 {rm.mean():.3f}（训练标签均值 {tiers.mean():.3f}）")

# ---------------- B
curve = {}
with open(os.path.join(R, "e3_comprehensive", "val", "curve_tree.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        curve.setdefault((r["policy"], r["tree"]), []).append((float(r["pruning_ratio"]), float(r["kpi_sum"])))


def kpi_at(pol, tree, x):
    pts = sorted(curve[(pol, tree)])
    xs = np.r_[0.0, [p[0] for p in pts]]
    ys = np.r_[0.0, [p[1] for p in pts]]
    return float(np.interp(x, xs, ys))


print("\nB. 同剪枝量比较：各策略在 DT(bias −0.6) 各树停止剪枝比处的 KPI 综合（曲线线性插值，含 0 点）")
dt_end = np.array([ksum(m06[n]) for n in trees])
dt_curve_at = np.array([kpi_at("dt", n, x) for n, x in zip(trees, rm)])
print(f"   DT 终点 KPI {dt_end.mean():.1f}；DT 受控曲线在同一剪枝比处 {dt_curve_at.mean():.1f}（插值误差参照）")
for pol in ["expert", "bc", "bcq", "cql", "iql", "random", "greedy"]:
    o = np.array([kpi_at(pol, n, x) for n, x in zip(trees, rm)])
    d = dt_end - o
    p = wilcoxon(dt_end, o).pvalue if np.any(d != 0) else 1
    print(f"   DT − {pol:<7} {d.mean():+7.1f}  胜率 {np.mean(d > 0):.0%}  p={p:.3g}")

# ---------------- C
print("\nC. 各策略终点剪枝比分布（e3_comprehensive 终点口径；DT 取 bias −0.6）")
srcs = {"dt": ("stop_diag", "dt_bias_m06")}
for p in ["expert", "bc", "bcq", "cql", "iql", "random", "greedy"]:
    srcs[p] = ("e3_v2", p)
for p, (e, t) in srcs.items():
    d = load(e, t)
    r = np.array([float(d[n]["final_pruning_ratio"]) for n in trees])
    print(f"   {p:<7} 均值 {r.mean():.3f}  <0.18 {np.mean(r < 0.18):.0%}  <0.25 {np.mean(r < 0.25):.0%}  "
          f"[0.18,0.32] {np.mean((r >= 0.18) & (r <= 0.32)):.0%}  ≥0.295（触顶）{np.mean(r >= 0.295):.0%}")

# ---------------- E
tgt = np.array([float(base[n]["target_return"]) for n in trees])
steps = np.array([float(base[n]["steps"]) for n in trees])
print("\nE. DT 停止位置与树规模（目标回报 = 系数 × 初始分枝数，作规模代理）")
print(f"   未校准：剪枝比与规模 Spearman {spearmanr(rb, tgt)[0]:+.2f}；bias −0.6：{spearmanr(rm, tgt)[0]:+.2f}")
order = np.argsort(tgt)
for name, idx in zip(["小树", "中树", "大树"], np.array_split(order, 3)):
    print(f"   {name}（{len(idx)} 棵，目标均值 {tgt[idx].mean():.0f}）：未校准剪枝比 {rb[idx].mean():.3f}，bias −0.6 {rm[idx].mean():.3f}，"
          f"专家 {np.mean([float(load('e3_v2', 'expert')[trees[i]]['final_pruning_ratio']) for i in idx]):.3f}")
