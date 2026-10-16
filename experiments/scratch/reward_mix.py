"""新奖励下各分项占比（训练树，专家 vs 随机）：α·R_q、β·R_i（及六项）、γ·R_c、终止奖励。"""
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
acc = {}
for n in load_val_split()["train"]:
    for r in load_parts(n):
        p, dk = r["parts"], r["dk"]
        ri_k = np.clip(dk / s, -1, 1) * w
        d = acc.setdefault(r["policy"], {"rq": [], "prot": [], "ri": [], "rc": [], "stop": [], "k": [],
                                         "steps": [], "ri_step": [], "rq_step": []})
        d["rq"].append(cfg.alpha * p[:, 0].sum())
        d["prot"].append(cfg.alpha * p[:, 1].sum())
        d["ri"].append(cfg.beta * ri_k.sum())
        d["rc"].append(cfg.gamma * p[:, 2].sum())
        d["stop"].append(r["stop"])
        d["k"].append(cfg.beta * ri_k.sum(0))
        d["steps"].append(len(p))
        d["ri_step"].append(ri_k.sum(1))
        d["rq_step"].append(p[:, 0] + p[:, 1])

print(f"α={cfg.alpha} β={cfg.beta} γ={cfg.gamma}")
for pol, d in acc.items():
    tot = np.array(d["rq"]) + np.array(d["prot"]) + np.array(d["ri"]) + np.array(d["rc"]) + np.array(d["stop"])
    print(f"\n[{pol}] {len(tot)} 条，步数均值 {np.mean(d['steps']):.1f}，总回报均值 {tot.mean():.1f}")
    for key, lab in (("rq", "α·R_q 不良枝"), ("prot", "α·保护惩罚"), ("ri", "β·R_i"),
                     ("rc", "γ·R_c"), ("stop", "终止")):
        print(f"  {lab:<12} {np.mean(d[key]):8.1f}")
    km = np.mean(d["k"], axis=0)
    print("  β·R_i 分项：" + "  ".join(f"{k} {v:.1f}" for k, v in zip(KPI_KEYS, km)))
    ri, rq = np.concatenate(d["ri_step"]), np.concatenate(d["rq_step"])
    print(f"  单刀 R_i 均值 {ri.mean():.2f}，>0 占比 {np.mean(ri > 0):.1%}；与 R_q 相关 {np.corrcoef(ri, rq)[0, 1]:.2f}")
