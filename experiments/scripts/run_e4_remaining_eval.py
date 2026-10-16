"""评估 E4 剩余两个核心消融，只跑验证树。

每个变体两次：自主停止，以及屏蔽 STOP 直至剪枝比 0.30。
去园艺知识使用 norq 回报，并从随机轨迹计算目标回报。
去安全约束奖励使用 norc 回报，目标回报仍来自专家轨迹。

结果写入 experiments/results/dev/e4_ablation/val/。不跑测试树。

用法（项目根目录）：
    py -3.12 experiments/scripts/run_e4_remaining_eval.py --workers 8
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

# (tag, checkpoint, reward_set, target_policy, min_pruning_ratio)
JOBS = [
    ("noknowledge", "checkpoints_dt_noknowledge/dt_best.pt", "norq", "random", 0.0),
    ("curve_noknowledge", "checkpoints_dt_noknowledge/dt_best.pt", "norq", "random", 0.30),
    ("norc", "checkpoints_dt_norc/dt_best.pt", "norc", "expert", 0.0),
    ("curve_norc", "checkpoints_dt_norc/dt_best.pt", "norc", "expert", 0.30),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="冒烟测试：仅跑前 N 棵")
    ap.add_argument("--exp", default="e4_ablation")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    result_dir = os.path.join(ROOT, "experiments", "results", "dev", args.exp, "val")
    os.makedirs(result_dir, exist_ok=True)
    started = time.time()
    for tag, checkpoint, reward_set, target_policy, min_ratio in JOBS:
        checkpoint_path = os.path.join(ROOT, "experiments", checkpoint)
        if not os.path.exists(checkpoint_path):
            raise SystemExit(f"缺少 {checkpoint_path}，请先完成训练")
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
            "--target-policy", target_policy,
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
    print(f"\n评估完成，累计 {(time.time() - started) / 60:.1f} min", flush=True)
    print("汇总命令：py -3.12 experiments/scripts/e4_summary.py")


if __name__ == "__main__":
    main()
