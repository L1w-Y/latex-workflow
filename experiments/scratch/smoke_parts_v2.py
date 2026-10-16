"""reward_parts v2 冒烟：前 2 棵树重算，R_q / R_c 与旧分项文件比对；看 ΔK 量级。"""
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np

DT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning"))
sys.path.insert(0, DT)

from graph_cache import available_train_names  # noqa: E402
from reward import KPI_KEYS  # noqa: E402
from reward_parts import PARTS_ROOT  # noqa: E402

names = available_train_names()[:2]
bak = os.path.join(os.path.dirname(__file__), "parts_v1_bak")
os.makedirs(bak, exist_ok=True)
old = {}
for n in names:
    src = os.path.join(PARTS_ROOT, f"{n}.pkl")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(bak, f"{n}.pkl"))
    with open(os.path.join(bak, f"{n}.pkl"), "rb") as f:
        old[n] = pickle.load(f)

subprocess.run([sys.executable, "reward_parts.py", "compute", "--max-trees", "2", "--workers", "2"],
               cwd=DT, check=True)

for n in names:
    with open(os.path.join(PARTS_ROOT, f"{n}.pkl"), "rb") as f:
        new = pickle.load(f)
    for o, w in zip(old[n], new):
        assert o["policy"] == w["policy"]
        po, pn = o["parts"], w["parts"]
        assert np.allclose(po[:, 0], pn[:, 0], atol=1e-4), "rq_bad 不一致"
        assert np.allclose(po[:, 1], pn[:, 1], atol=1e-4), "rq_protect 不一致"
        assert np.allclose(po[:, 3], pn[:, 2], atol=1e-4), "rc 不一致"
        assert abs(o["stop"] - w["stop"]) < 1e-6
    dk = np.abs(np.concatenate([w["dk"] for w in new]))
    print(n, "R_q/R_c/终止与旧一致；|ΔK| 中位",
          {k: f"{np.median(dk[:, i]):.1e}" for i, k in enumerate(KPI_KEYS)},
          "95分位", {k: f"{np.quantile(dk[:, i], 0.95):.1e}" for i, k in enumerate(KPI_KEYS)})
