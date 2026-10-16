"""训练并验证 DT STOP 头微调候选。

训练三个 continue_weight（1/2/4），每个保存 epoch 2/4/6；随后只在验证集
评估九个候选并按预设约束汇总。长时间任务由用户执行。

输出：
- checkpoints: experiments/checkpoints_dt_stopcal/w*/
- results: experiments/results/dev/stop_finetune/val/

用法（项目根目录）：
    python experiments/scripts/run_stop_finetune.py --workers 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TRAIN = os.path.join(ROOT, "dt_pruning", "train_stop.py")
EVALUATE = os.path.join(ROOT, "dt_pruning", "evaluate.py")
SUMMARY = os.path.join(HERE, "stop_finetune_summary.py")


def weight_tag(weight: float) -> str:
    return str(weight).replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--weights", default="1,2,4")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=2)
    ap.add_argument("--batches-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-trees", type=int, default=None, help="冒烟测试")
    args = ap.parse_args()
    if args.train_only and args.eval_only:
        raise SystemExit("--train-only 与 --eval-only 不能同时使用")

    weights = [float(value) for value in args.weights.split(",") if value]
    epochs = list(range(args.save_every, args.epochs + 1, args.save_every))
    if args.epochs not in epochs:
        epochs.append(args.epochs)
    checkpoint_root = (
        "scratch/checkpoints_dt_stopcal_smoke"
        if args.max_trees else "checkpoints_dt_stopcal"
    )
    eval_exp = "stop_finetune_smoke" if args.max_trees else "stop_finetune"
    started = time.time()

    if not args.eval_only:
        for weight in weights:
            tag = weight_tag(weight)
            out = f"{checkpoint_root}/w{tag}"
            final_path = os.path.join(
                ROOT, "experiments", out, f"dt_stop_ep{args.epochs}.pt"
            )
            if os.path.exists(final_path) and not args.overwrite:
                print(f"[train w={weight:g}] 已有最终 checkpoint，跳过", flush=True)
                continue
            cmd = [
                sys.executable, TRAIN,
                "--out", out,
                "--continue-weight", str(weight),
                "--epochs", str(args.epochs),
                "--save-every", str(args.save_every),
                "--batches-per-epoch", str(args.batches_per_epoch),
                "--batch-size", str(args.batch_size),
            ]
            if args.max_trees:
                cmd += ["--max-trees", str(args.max_trees)]
            print("\n" + "=" * 72)
            print(f"[train w={weight:g}] {' '.join(cmd)}")
            print("=" * 72, flush=True)
            subprocess.run(cmd, check=True, cwd=os.path.join(ROOT, "dt_pruning"))
            print(
                f"[train w={weight:g}] 累计用时 {(time.time() - started) / 60:.1f} min",
                flush=True,
            )

    if args.train_only:
        return

    result_dir = os.path.join(
        ROOT, "experiments", "results", "dev", eval_exp, "val"
    )
    for weight in weights:
        wtag = weight_tag(weight)
        for epoch in epochs:
            tag = f"dt_w{wtag}_ep{epoch}"
            csv_path = os.path.join(result_dir, f"eval_{tag}.csv")
            actions_path = os.path.join(result_dir, f"actions_{tag}.json")
            if (os.path.exists(csv_path) and os.path.exists(actions_path)
                    and not args.overwrite):
                print(f"[{tag}] 已有完整结果，跳过", flush=True)
                continue
            checkpoint = f"{checkpoint_root}/w{wtag}/dt_stop_ep{epoch}.pt"
            absolute = os.path.join(ROOT, "experiments", checkpoint)
            if not os.path.exists(absolute):
                raise SystemExit(f"缺少 checkpoint：{absolute}")
            cmd = [
                sys.executable, EVALUATE,
                "--policy", "dt",
                "--checkpoint", checkpoint,
                "--split", "val",
                "--stage", "dev",
                "--exp", eval_exp,
                "--tag", tag,
                "--workers", str(args.workers),
            ]
            if args.max_trees:
                cmd += ["--limit", str(max(1, args.max_trees // 3))]
            print("\n" + "=" * 72)
            print(f"[{tag}] {' '.join(cmd)}")
            print("=" * 72, flush=True)
            subprocess.run(cmd, check=True, cwd=os.path.join(ROOT, "dt_pruning"))
            print(
                f"[{tag}] 累计用时 {(time.time() - started) / 60:.1f} min",
                flush=True,
            )

    if args.max_trees:
        print("\n冒烟测试完成；--max-trees 结果不执行候选汇总。")
        return
    subprocess.run(
        [sys.executable, SUMMARY, "--exp", eval_exp],
        check=True,
        cwd=ROOT,
    )
    print(f"\nSTOP 微调与验证完成，总用时 {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
