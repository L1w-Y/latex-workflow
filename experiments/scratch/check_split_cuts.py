"""检查"先剪子枝、后剪其祖先枝"的比例：同一块材料被拆成多刀分别计奖励的程度。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning"))
from config import CFG  # noqa: E402
from tree_data import build_parent_map, load_skeleton, skeleton_path  # noqa: E402

exp = sys.argv[1] if len(sys.argv) > 1 else "e2_teacher"
pols = sys.argv[2].split(",") if len(sys.argv) > 2 else ["dt", "expert", "bc"]
d = os.path.join(os.path.dirname(__file__), "..", "results", "dev", exp, "val")
acts = {p: json.load(open(os.path.join(d, f"actions_{p}.json"), encoding="utf-8")) for p in pols}
pm_cache = {}
out = {p: {"cuts": 0, "split": 0, "split_reward": 0.0, "reward": 0.0, "single": 0} for p in acts}
for tree in acts["dt"]:
    if tree not in pm_cache:
        pm_cache[tree] = build_parent_map(load_skeleton(skeleton_path(os.path.join(CFG.data.data_root, tree))))
    pm = pm_cache[tree]
    for p, a in acts.items():
        trace = a[tree]
        later = set()
        for c in reversed(trace):
            b, anc, x = c["bid"], set(), c["bid"]
            while pm.get(x, -1) >= 0:
                x = pm[x]
                anc.add(x)
            o = out[p]
            o["cuts"] += 1
            o["reward"] += c["reward"]
            o["single"] += len(c["removed"]) == 1
            if anc & later:
                o["split"] += 1
                o["split_reward"] += c["reward"]
            later.add(b)
for p, o in out.items():
    print(p, "cuts", o["cuts"], "child-before-ancestor rate %.3f" % (o["split"] / o["cuts"]),
          "reward share %.3f" % (o["split_reward"] / o["reward"]),
          "leaf-only cut rate %.3f" % (o["single"] / o["cuts"]))
