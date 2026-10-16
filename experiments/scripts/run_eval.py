"""一键评估：DT / BC / 贪心 / 专家 / 随机 在同一批树上对比 + 配对 Wilcoxon 检验。

逐树结果落盘到 experiments/results/<stage>/<exp>/<split>/eval_<policy>.csv，
检验结果在同目录 significance_dt.json。

用法（在项目根目录执行）：
    python experiments/scripts/run_eval.py --split val --exp tune_x --workers 8                 # 调参：验证树，写入 results/dev/
    python experiments/scripts/run_eval.py --split test --stage paper --exp main --workers 8   # 最终报告：测试树，写入 results/paper/

参数：
    --split        val（调参用）/ test（只在定稿后跑，必须 --stage paper）
    --stage        dev（默认）/ paper
    --exp          实验子目录名；paper 阶段必填
    --limit        评估的树数量（所有策略跑同一批树，保证可配对）
    --dt / --bc    checkpoint 路径（相对 experiments/）；传空字符串跳过该策略
    --dt-args      透传给 DT 的额外参数，如 "--target-return 4.0"（默认按已定规则：per_branch、分位数 1.0）
    --offline      离线强化学习基线，逗号分隔（如 cql,bcq,iql），checkpoint 取 checkpoints_<algo>/<algo>_last.pt
    --policies     要跑的基线，逗号分隔（默认 greedy,expert,random）
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_DT = os.path.abspath(os.path.join(_HERE, "..", "..", "dt_pruning"))
_EVALUATE = os.path.join(_DT, "evaluate.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--stage", choices=["dev", "paper"], default="dev")
    ap.add_argument("--exp", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dt", type=str, default="checkpoints_dt/dt_best.pt")
    ap.add_argument("--dt-args", type=str, default="")
    ap.add_argument("--bc", type=str, default="checkpoints_bc/bc_best.pt")
    ap.add_argument("--offline", type=str, default="")
    ap.add_argument("--policies", type=str, default="greedy,expert,random")
    args = ap.parse_args()

    common = ["--split", args.split, "--stage", args.stage, "--workers", str(args.workers)]
    if args.exp:
        common += ["--exp", args.exp]
    if args.limit:
        common += ["--limit", str(args.limit)]
    jobs = []
    if args.dt:
        jobs.append(("dt", ["--checkpoint", args.dt] + shlex.split(args.dt_args)))
    if args.bc:
        jobs.append(("bc", ["--checkpoint", args.bc]))
    jobs += [(a, ["--checkpoint", f"checkpoints_{a}/{a}_last.pt"]) for a in args.offline.split(",") if a]
    jobs += [(p, []) for p in args.policies.split(",") if p]

    py = sys.executable
    t0 = time.time()
    for policy, extra in jobs:
        cmd = [py, _EVALUATE, "--policy", policy] + common + extra
        print("\n" + "=" * 70 + f"\n[{policy}]  " + " ".join(cmd) + "\n" + "=" * 70, flush=True)
        subprocess.run(cmd, check=True, cwd=_DT)
        print(f"[{policy}] 累计用时 {(time.time() - t0) / 60:.1f} min", flush=True)

    print("\n" + "=" * 70 + "\n[compare] 配对 Wilcoxon 检验\n" + "=" * 70, flush=True)
    cmp = [py, _EVALUATE, "--compare", "--split", args.split, "--stage", args.stage, "--ref", "dt"]
    if args.exp:
        cmp += ["--exp", args.exp]
    subprocess.run(cmp, check=True, cwd=_DT)
    print(f"\n全部完成，总用时 {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
