"""E3 各策略单刀画像：每刀规则分、枝长、半径、剪除点数、不良类别占比、无不良标签（功能枝 / 普通枝）占比。"""
import json
import os
from collections import Counter

import numpy as np

D = os.path.join(os.path.dirname(__file__), "..", "results", "dev", "e3_f5", "val")
for pol in ["dt", "expert", "bcq", "cql", "iql", "random"]:
    with open(os.path.join(D, f"actions_{pol}.json"), encoding="utf-8") as f:
        cuts = [c for tr in json.load(f).values() for c in tr]
    tags = Counter(t for c in cuts for t in c["tags"])
    n = len(cuts)
    print(f"{pol:<7} 刀 {n:5d}  R_q/刀 {np.mean([c['rq'] for c in cuts]):.2f}  R_i/刀 {np.mean([c['ri'] for c in cuts]):.2f}"
          f"  枝长 {np.median([c['length'] for c in cuts]):.2f}  半径 {np.median([c['radius'] for c in cuts]) * 1000:.1f}mm"
          f"  点数 中位 {np.median([c['points_removed'] for c in cuts]):.0f} / 90分位 {np.percentile([c['points_removed'] for c in cuts], 90):.0f}"
          f"  R_q≤0 占 {np.mean([c['rq'] <= 0 for c in cuts]):.0%}")
    print("        " + "  ".join(f"{k} {v / n:.0%}" for k, v in tags.most_common()))
