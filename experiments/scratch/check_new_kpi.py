"""新 KPI 接入核对：与原型取值比对 + 单次计算耗时 + 几步重放。"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))

from config import CFG  # noqa: E402
from evaluate import _Env, result_dir  # noqa: E402
from reward import compute_kpis  # noqa: E402

tree = sys.argv[1] if len(sys.argv) > 1 else "Lemon00_01"
env = _Env(os.path.join(CFG.data.data_root, tree), CFG.eval.seed)
print("剪前", {k: round(v, 4) for k, v in env.k.items()})
t0 = time.time()
for _ in range(3):
    compute_kpis(env.state.skeleton, env.props, CFG.reward, env.light_ctx)
print(f"compute_kpis 每次 {(time.time() - t0) / 3:.3f} s")
t0 = time.time()
for _ in range(3):
    env.light_ctx.light(env.props, CFG.reward)
print(f"其中光照每次 {(time.time() - t0) / 3:.3f} s")

acts = json.load(open(os.path.join(result_dir("dev", "e3_baselines", "val"), "actions_expert.json"), encoding="utf-8"))
k_first = dict(env.k)
for c in acts[tree][:20]:
    env.step(c["bid"])
print("专家 20 刀后", {k: round(env.k[k] - k_first[k], 4) for k in env.k if k != "branch_count"})
print("单刀 R_i", [round(c["ri"], 2) for c in env.trace[:10]])
