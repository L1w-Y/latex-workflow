"""同等剪除量下的 KPI 效率：累计 KPI 综合（ΣR_i）在剪除点数占比 0.10 / 0.15 / 0.20 处的值（e3_v2，逐树插值后取均值）。"""
import csv
import json
import os

import numpy as np

D = os.path.join(os.path.dirname(__file__), "..", "results", "dev", "e3_v2", "val")
GRID = [0.10, 0.15, 0.20, 0.23]
for pol in ["dt", "expert", "bcq", "cql", "iql", "random"]:
    with open(os.path.join(D, f"actions_{pol}.json"), encoding="utf-8") as f:
        tr = json.load(f)
    with open(os.path.join(D, f"eval_{pol}.csv"), encoding="utf-8") as f:
        ratio = {r["tree"]: float(r["final_pruning_ratio"]) for r in csv.DictReader(f)}
    vals = {g: [] for g in GRID}
    for n, cuts in tr.items():
        pts = np.cumsum([c["points_removed"] for c in cuts])
        if not len(pts):
            continue
        frac = pts / pts[-1] * ratio[n]          # 剪除点数占比 ≈ 剪枝比（按终点对齐）
        ki = np.cumsum([c["ri"] for c in cuts])
        for g in GRID:
            if frac[-1] >= g:
                vals[g].append(np.interp(g, np.r_[0, frac], np.r_[0, ki]))
    print(f"{pol:<7}" + "  ".join(f"@{g:.2f}: {np.mean(v):6.1f}（{len(v)} 棵）" for g, v in vals.items()))
