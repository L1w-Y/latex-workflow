"""修订奖励（R_q 按量 + R_i 不截断）下训练树各分项：专家 vs 随机。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))
from config import CFG  # noqa: E402
from graph_cache import load_val_split  # noqa: E402
from reward import KPI_KEYS  # noqa: E402
from reward_parts import load_parts, step_rewards  # noqa: E402

cfg = CFG.reward
w = np.array([cfg.kpi_weights[k] for k in KPI_KEYS])
s = np.array([cfg.kpi_scales[k] for k in KPI_KEYS])
acc = {}
for n in load_val_split()["train"]:
    for r in load_parts(n):
        p, sc = r["parts"], r["pts"] / cfg.rq_volume_ref
        d = acc.setdefault(r["policy"], {k: [] for k in ("tot", "rq", "prot", "ri", "rc", "stop")})
        d["tot"].append(step_rewards(r).sum())
        d["rq"].append(cfg.alpha * (p[:, 0] * sc).sum())
        d["prot"].append(cfg.alpha * (p[:, 1] * sc).sum())
        d["ri"].append(cfg.beta * (r["dk"] / s * w).sum())
        d["rc"].append(cfg.gamma * p[:, 2].sum())
        d["stop"].append(r["stop"])
for pol, d in acc.items():
    m = {k: np.mean(v) for k, v in d.items()}
    print(f"[{pol}] {len(d['tot'])} 条  总回报 {m['tot']:.0f} = 规则分 {m['rq']:.0f} + 保护惩罚 {m['prot']:.0f}"
          f" + KPI 分 {m['ri']:.0f} + 约束 {m['rc']:.1f} + 终止 {m['stop']:.1f}；KPI 分占比 {m['ri'] / m['tot']:.0%}")
