"""运行 DT STOP 验证集诊断，并在全部完成后自动汇总。

默认运行四组：
1. 原始自主停止；
2. 初始目标回报 ×1.25；
3. 达到剪枝比 0.26 前屏蔽 STOP；
4. 达到剪枝比 0.30 前屏蔽 STOP。

用法（项目根目录）：
    python experiments/scripts/run_stop_diag.py --workers 8
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

    variants = [
        ("dt_base", []),
        ("dt_target125", ["--target-multiplier", "1.25"]),
        ("dt_min026", ["--min-pruning-ratio", "0.26"]),
        ("dt_min030", ["--min-pruning-ratio", "0.30"]),
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
    print(f"\n诊断完成，总用时 {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
