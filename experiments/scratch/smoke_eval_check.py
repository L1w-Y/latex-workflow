"""冒烟评估（1 棵）结果检查：各策略剪后 KPI 变化、刀数、剪枝比、回报拆分。"""
import csv
import glob
import json
import os
import sys

csv.field_size_limit(10 ** 9)
D = os.path.join(os.path.dirname(__file__), "..", "results", "dev", "smoke", "val")
KEYS = ["hierarchy", "crowding", "balance", "light", "potential", "stability"]
print(f"{'policy':<8}{'ret':>8}{'cuts':>6}{'ratio':>7}{'5Rq':>8}{'3Ri':>8}{'Rc':>7}  " + " ".join(f"{k[:5]:>7}" for k in KEYS))
for p in sorted(glob.glob(os.path.join(D, "eval_*.csv"))):
    with open(p.replace("eval_", "actions_").replace(".csv", ".json"), encoding="utf-8") as f:
        traces = json.load(f)
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tr = traces[row["tree"]]
            rq = sum(t.get("rq", 0) for t in tr) * 5
            ri = sum(t.get("ri", 0) for t in tr) * 3
            rc = sum(t.get("rc", 0) for t in tr)
            print(f"{os.path.basename(p)[5:-4]:<8}{float(row['total_reward']):8.0f}{int(row['steps']):6d}"
                  f"{float(row['final_pruning_ratio']):7.3f}{rq:8.0f}{ri:8.0f}{rc:7.1f}  "
                  + " ".join(f"{float(row[k + '_delta']):+7.3f}" for k in KEYS))
