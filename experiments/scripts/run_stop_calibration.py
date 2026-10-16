"""继续运行 DT STOP 推理校准实验，并与已有 stop_diag 结果合并汇总。

默认新增四组：
- 初始目标回报 ×1.5 / ×2.0
- STOP logit 偏置 -0.2 / -0.6

先运行 experiments/scripts/run_stop_diag.py 生成 dt_base，再运行本脚本。

用法（项目根目录）：
    python experiments/scripts/run_stop_calibration.py --workers 8
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
SUMMARY = os.path.join(HERE, "stop_diagnosis.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="stop_diag")
    ap.add_argument("--checkpoint", default="checkpoints_dt/dt_best.pt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    result_dir = os.path.join(ROOT, "experiments", "results", "dev",
                              args.exp, "val")
    if not os.path.exists(os.path.join(result_dir, "eval_dt_base.csv")):
        raise SystemExit(
            f"缺少 {result_dir}\\eval_dt_base.csv；请先运行 run_stop_diag.py"
        )

    variants = [
        ("dt_target150", ["--target-multiplier", "1.5"]),
        ("dt_target200", ["--target-multiplier", "2.0"]),
        ("dt_bias_m02", ["--stop-logit-bias", "-0.2"]),
        ("dt_bias_m06", ["--stop-logit-bias", "-0.6"]),
    ]
    common = [
        "--policy", "dt",
        "--checkpoint", args.checkpoint,
        "--split", "val",
        "--stage", "dev",
        "--exp", args.exp,
        "--workers", str(args.workers),
    ]
    if args.limit:
        common += ["--limit", str(args.limit)]

    started = time.time()
    for tag, extra in variants:
        cmd = [sys.executable, EVALUATE, *common, "--tag", tag, *extra]
        print("\n" + "=" * 72)
        print(f"[{tag}] {' '.join(cmd)}")
        print("=" * 72, flush=True)
        subprocess.run(cmd, check=True, cwd=os.path.join(ROOT, "dt_pruning"))
        print(f"[{tag}] 累计用时 {(time.time() - started) / 60:.1f} min", flush=True)

    subprocess.run(
        [sys.executable, SUMMARY, "--exp", args.exp, "--ref", "dt_base"],
        check=True,
        cwd=ROOT,
    )
    print(f"\n校准实验完成，总用时 {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
