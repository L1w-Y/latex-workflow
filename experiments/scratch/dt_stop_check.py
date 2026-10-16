"""DT 提前停手排查：逐树目标回报 vs 实得回报、剪枝比、刀数，与专家对照；e3_f5 与 e3_v2 两轮。"""
import csv
import json
import os

import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results", "dev")


def rows(exp, pol):
    with open(os.path.join(R, exp, "val", f"eval_{pol}.csv"), encoding="utf-8") as f:
        return {r["tree"]: r for r in csv.DictReader(f)}


for exp in ["e3_f5", "e3_v2"]:
    dt, ex = rows(exp, "dt"), rows(exp, "expert")
    t = sorted(dt)
    tgt = np.array([float(dt[n]["target_return"]) for n in t])
    got = np.array([float(dt[n]["total_reward"]) for n in t])
    ratio = np.array([float(dt[n]["final_pruning_ratio"]) for n in t])
    eret = np.array([float(ex[n]["total_reward"]) for n in t])
    eratio = np.array([float(ex[n]["final_pruning_ratio"]) for n in t])
    print(f"\n[{exp}] 目标回报均值 {tgt.mean():.0f}，DT 实得 {got.mean():.0f}（达成 {np.mean(got / tgt):.0%}），专家 {eret.mean():.0f}")
    print(f"  剪枝比 DT {ratio.mean():.3f} / 专家 {eratio.mean():.3f}；DT 剪枝比 < 0.25 的树 {np.sum(ratio < 0.25)} 棵，≥ 0.29 的 {np.sum(ratio >= 0.29)} 棵")
    print(f"  DT 实得 / 目标 与 剪枝比 相关 {np.corrcoef(got / tgt, ratio)[0, 1]:.2f}；目标 / 专家回报 中位 {np.median(tgt / eret):.2f}")
    with open(os.path.join(R, exp, "val", "actions_dt.json"), encoding="utf-8") as f:
        tr = json.load(f)
    # 剪枝比分布：低剪枝比的树是否最后几刀回报变低、或是在第几刀停
    lo = [n for n, r in zip(t, ratio) if r < 0.25]
    if lo:
        print("  低剪枝比树示例：" + "；".join(
            f"{n} 比 {float(dt[n]['final_pruning_ratio']):.2f} 刀 {len(tr[n])} 实得/目标 {float(dt[n]['total_reward']):.0f}/{float(dt[n]['target_return']):.0f}"
            for n in lo[:5]))
