"""运行 E3 验证集综合基线的过程评估并自动汇总。

用途仅限验证集开发阶段，结果写入：
    experiments/results/dev/e3_comprehensive/val/

终点指标复用：
    dev/e3_v2/val/（各基线）
    dev/stop_diag/val/eval_dt_bias_m06.*（暂定 DT）

用法（项目根目录）：
    python experiments/scripts/run_e3_comprehensive.py --workers 8
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
ANALYZE = os.path.join(HERE, "e3_comprehensive.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="冒烟测试：仅跑前 N 棵")
    ap.add_argument("--exp", default="e3_comprehensive")
    ap.add_argument("--overwrite", action="store_true", help="重跑并覆盖已完成的过程结果")
    args = ap.parse_args()

    jobs = [
        ("dt", "checkpoints_dt/dt_best.pt",
         ["--min-pruning-ratio", "0.30", "--stop-logit-bias", "-0.6"]),
        ("bc", "checkpoints_bc/bc_best.pt", ["--min-pruning-ratio", "0.30"]),
        ("bcq", "checkpoints_bcq/bcq_last.pt", ["--min-pruning-ratio", "0.30"]),
        ("cql", "checkpoints_cql/cql_last.pt", ["--min-pruning-ratio", "0.30"]),
        ("iql", "checkpoints_iql/iql_last.pt", ["--min-pruning-ratio", "0.30"]),
        ("expert", None, []),
        ("greedy", None, []),
        ("random", None, []),
    ]
    common = [
        "--split", "val",
        "--stage", "dev",
        "--exp", args.exp,
        "--workers", str(args.workers),
    ]
    if args.limit:
        common += ["--limit", str(args.limit)]

    started = time.time()
    for policy, checkpoint, extra in jobs:
        result_dir = os.path.join(
            ROOT, "experiments", "results", "dev", args.exp, "val"
        )
        csv_path = os.path.join(result_dir, f"eval_curve_{policy}.csv")
        actions_path = os.path.join(result_dir, f"actions_curve_{policy}.json")
        if not args.overwrite and os.path.exists(csv_path) and os.path.exists(actions_path):
            print(f"[curve_{policy}] 已有完整结果，跳过；重跑请加 --overwrite", flush=True)
            continue
        cmd = [
            sys.executable, EVALUATE,
            "--policy", policy,
            *common,
            "--tag", f"curve_{policy}",
        ]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        cmd += extra
        print("\n" + "=" * 72)
        print(f"[curve_{policy}] {' '.join(cmd)}")
        print("=" * 72, flush=True)
        subprocess.run(cmd, check=True, cwd=os.path.join(ROOT, "dt_pruning"))
        print(
            f"[curve_{policy}] 累计用时 {(time.time() - started) / 60:.1f} min",
            flush=True,
        )

    if args.limit:
        print("\n冒烟测试完成；--limit 结果不执行综合汇总。")
        return

    cmd = [
        sys.executable, ANALYZE,
        "--stage", "dev",
        "--split", "val",
        "--endpoint-exp", "e3_v2",
        "--stop-exp", "stop_diag",
        "--out-exp", args.exp,
        "--dt-tag", "dt_bias_m06",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"\nE3 综合基线完成，总用时 {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
