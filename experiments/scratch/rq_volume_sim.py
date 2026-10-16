"""离线估算：R_q 按剪除量缩放（+ R_i 不逐刀截断）后，现有各策略的回报排名是否与 KPI 综合排名一致。

R_q' = R_q × 剪除点数 / c，c = 训练树专家轨迹每刀平均剪除点数（使总量与原 R_q 同量级）。
用 E3 验证树评估记录（每刀 rq、ri、points_removed）与剪后 KPI 综合。
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))
from config import CFG  # noqa: E402

KEYS = ["hierarchy", "crowding", "balance", "light", "potential", "stability"]
D = os.path.join(os.path.dirname(__file__), "..", "results", "dev", "e3_f5", "val")
LEARNED = ["dt", "bc", "bcq", "cql", "iql"]
cfg = CFG.reward

data = {}
for pol in LEARNED + ["expert", "random"]:
    with open(os.path.join(D, f"actions_{pol}.json"), encoding="utf-8") as f:
        tr = json.load(f)
    with open(os.path.join(D, f"eval_{pol}.csv"), encoding="utf-8") as f:
        rows = {r["tree"]: r for r in csv.DictReader(f)}
    data[pol] = {n: (tr[n], sum(cfg.kpi_weights[k] * float(rows[n][f"{k}_delta"]) / cfg.kpi_scales[k] for k in KEYS),
                     float(rows[n]["total_reward"]) - sum(c["reward"] for c in tr[n]))
                 for n in rows}
c = np.mean([cut["points_removed"] for tr, _, _ in data["expert"].values() for cut in tr])
print(f"c = 专家每刀平均剪除点数 {c:.1f}（验证树近似）\n")


def ret(tr, ksum, stop, beta, vol, clip, cap=1e9):
    def s(cut):
        if not vol:
            return 1.0
        if cap == "main1" and "main_branch" in cut["tags"]:
            return 1.0
        return min(cut["points_removed"] / c, cap if cap != "main1" else 1e9)
    rq = sum(cut["rq"] * s(cut) for cut in tr)
    ri = sum(cut["ri"] for cut in tr) if clip else ksum
    rc = sum(cut["rc"] for cut in tr)
    return cfg.alpha * rq + beta * ri + cfg.gamma * rc + stop


kpi = {p: np.mean([v[1] for v in d.values()]) for p, d in data.items()}
print("KPI 综合：" + "  ".join(f"{p} {kpi[p]:.0f}" for p in data))
for pol in ["dt", "expert", "iql"]:
    s = np.array([cut["points_removed"] / c for tr, _, _ in data[pol].values() for cut in tr])
    mb = [cut["points_removed"] / c for tr, _, _ in data[pol].values() for cut in tr if "main_branch" in cut["tags"]]
    print(f"{pol} 缩放系数分位 50/90/99: {np.percentile(s, [50, 90, 99]).round(2)}；骨干枝刀 {len(mb)} 次，系数中位 {np.median(mb) if mb else 0:.1f}")
for beta, vol, clip, cap, lab in [(3, False, True, 1e9, "现行"),
                                  (3, True, False, 1e9, "R_q 按量 + R_i 不截断"),
                                  (3, True, False, 5, "同上，系数上限 5"),
                                  (3, True, False, 3, "同上，系数上限 3"),
                                  (3, True, False, "main1", "同上，骨干枝不放大（系数 1）")]:
    R = {p: np.mean([ret(*v, beta, vol, clip, cap) for v in d.values()]) for p, d in data.items()}
    rho = spearmanr([R[p] for p in LEARNED], [kpi[p] for p in LEARNED])[0]
    # 树内一致性：同一棵树上学习方法之间，回报高者 KPI 是否也高
    agree = []
    for n in data["dt"]:
        for i, a in enumerate(LEARNED):
            for b in LEARNED[i + 1:]:
                ra, rb = ret(*data[a][n], beta, vol, clip, cap), ret(*data[b][n], beta, vol, clip, cap)
                ka, kb = data[a][n][1], data[b][n][1]
                if ra != rb and ka != kb:
                    agree.append((ra > rb) == (ka > kb))
    print(f"\n[{lab}] 学习方法回报排名与 KPI 排名 Spearman {rho:+.2f}；树内两两一致 {np.mean(agree):.0%}")
    print("  回报：" + "  ".join(f"{p} {R[p]:.0f}" for p in sorted(R, key=R.get, reverse=True)))
