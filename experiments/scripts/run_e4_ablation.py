"""按顺序运行 E4 五组消融的验证集评估。

每组两次：自主停止（停止偏置 0），以及屏蔽 STOP 直至剪枝比 0.30。
十次串行，不并行。已有 csv 与 actions 的任务跳过；某次失败则停止，重跑会接着未完成的做。

结果写入 experiments/results/dev/e4_ablation/val/。不跑测试树，不覆盖 checkpoints_dt。

用法（项目根目录）：
    py -3.12 experiments/scripts/run_e4_ablation.py --workers 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EVALUATE = os.path.join(ROOT, "dt_pruning", "evaluate.py")

# (tag, checkpoint, reward_set, min_pruning_ratio)
JOBS = [
    ("k1", "checkpoints_dt_k1/dt_best.pt", "full", 0.0),
    ("curve_k1", "checkpoints_dt_k1/dt_best.pt", "full", 0.30),
    ("expert", "checkpoints_dt_expert/dt_best.pt", "full", 0.0),
    ("curve_expert", "checkpoints_dt_expert/dt_best.pt", "full", 0.30),
    ("norq", "checkpoints_dt_norq/dt_best.pt", "norq", 0.0),
    ("curve_norq", "checkpoints_dt_norq/dt_best.pt", "norq", 0.30),
    ("nori", "checkpoints_dt_nori/dt_best.pt", "nori", 0.0),
    ("curve_nori", "checkpoints_dt_nori/dt_best.pt", "nori", 0.30),
    ("noprotect", "checkpoints_dt_noprotect/dt_best.pt", "noprotect", 0.0),
    ("curve_noprotect", "checkpoints_dt_noprotect/dt_best.pt", "noprotect", 0.30),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="冒烟测试：仅跑前 N 棵")
    ap.add_argument("--exp", default="e4_ablation")
    ap.add_argument("--overwrite", action="store_true", help="重跑并覆盖已有结果")
    args = ap.parse_args()

    result_dir = os.path.join(ROOT, "experiments", "results", "dev", args.exp, "val")
    os.makedirs(result_dir, exist_ok=True)
    started = time.time()
    for tag, checkpoint, reward_set, min_ratio in JOBS:
        csv_path = os.path.join(result_dir, f"eval_{tag}.csv")
        actions_path = os.path.join(result_dir, f"actions_{tag}.json")
        if not args.overwrite and os.path.exists(csv_path) and os.path.exists(actions_path):
            print(f"[{tag}] 已有完整结果，跳过；重跑请加 --overwrite", flush=True)
            continue
        cmd = [
            sys.executable, EVALUATE,
            "--policy", "dt",
            "--checkpoint", checkpoint,
            "--split", "val",
            "--stage", "dev",
            "--exp", args.exp,
            "--workers", str(args.workers),
            "--tag", tag,
            "--reward-set", reward_set,
            "--stop-logit-bias", "0",
            "--min-pruning-ratio", str(min_ratio),
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print("\n" + "=" * 72)
        print(f"[{tag}] {' '.join(cmd)}")
        print("=" * 72, flush=True)
        subprocess.run(cmd, check=True, cwd=os.path.join(ROOT, "dt_pruning"))
        print(f"[{tag}] 累计用时 {(time.time() - started) / 60:.1f} min", flush=True)
    print(f"\n全部完成，累计 {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
