"""单步奖励裁剪 [clip_min, clip_max] 的影响：不同 β 下被裁步占比、裁掉的回报、专家 / 随机差距。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))
from config import CFG  # noqa: E402
from graph_cache import load_val_split  # noqa: E402
from reward import KPI_KEYS  # noqa: E402
from reward_parts import load_parts  # noqa: E402

cfg = CFG.reward
w = np.array([cfg.kpi_weights[k] for k in KPI_KEYS])
s = np.array([cfg.kpi_scales[k] for k in KPI_KEYS])
recs = [r for n in load_val_split()["train"] for r in load_parts(n)]
for beta in (1.0, 3.0):
    for lo, hi in ((cfg.clip_min, cfg.clip_max), (cfg.clip_min, 1e9), (-1e9, 1e9)):
        out = {}
        for r in recs:
            p = r["parts"]
            ri = (np.clip(r["dk"] / s, -1, 1) * w).sum(1)
            raw = cfg.alpha * (p[:, 0] + p[:, 1]) + beta * ri + cfg.gamma * p[:, 2]
            c = np.clip(raw, lo, hi)
            d = out.setdefault(r["policy"], [[], [], [], []])
            d[0].append(c.sum())
            d[1].append(np.mean(raw > hi))
            d[2].append(np.mean(raw < lo))
            d[3].append(beta * ri.sum())
        e, rd = out["expert"], out["random"]
        print(f"β={beta} 裁剪[{lo:g},{hi:g}]  专家 {np.mean(e[0]):.0f}（超上限 {np.mean(e[1]):.1%}）"
              f"  随机 {np.mean(rd[0]):.0f}（超上限 {np.mean(rd[1]):.1%}，低于下限 {np.mean(rd[2]):.1%}）"
              f"  差距 {np.mean(e[0]) - np.mean(rd[0]):.0f}")
p = np.concatenate([r["parts"][:, 0] for r in recs if r["policy"] == "expert"])
print("专家单刀 R_q 分位 50/90/99:", np.percentile(p, [50, 90, 99]).round(2))
pr = np.concatenate([r["parts"][:, 1] for r in recs if r["policy"] == "random"])
print("随机保护惩罚（<0 的刀）分位 1/10/50:", np.percentile(pr[pr < 0], [1, 10, 50]).round(2))
