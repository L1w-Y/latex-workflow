"""训练 E4 剩余的两个核心消融，不重新生成轨迹。

顺序：
1. 检查并在缺失时生成 rtg_norc（只重组已有奖励分项）
2. 去园艺知识：只训练随机轨迹，回报集 norq，不加载专家 BC
3. 去安全约束奖励：专家与随机轨迹，回报集 norc，BC 编码器初始化

某个变体已有完整 60 epoch 日志和 dt_best.pt 时跳过。
中断后重跑会从头训练未完成的变体，不覆盖 checkpoints_dt。

用法（项目根目录）：
    py -3.12 experiments/scripts/run_e4_remaining_train.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DT = os.path.join(ROOT, "dt_pruning")
EXP = os.path.join(ROOT, "experiments")
EPOCHS = 60

JOBS = [
    (
        "去园艺知识",
        "checkpoints_dt_noknowledge",
        [
            "--policies", "random",
            "--expert-weight", "0",
            "--reward-set", "norq",
            "--epochs", str(EPOCHS),
            "--context-len", "16",
            "--batch-size", "8",
            "--rtg-scale", "1000",
            "--out", "checkpoints_dt_noknowledge",
        ],
    ),
    (
        "去安全约束奖励",
        "checkpoints_dt_norc",
        [
            "--policies", "expert,random",
            "--expert-weight", "0.5",
            "--reward-set", "norc",
            "--epochs", str(EPOCHS),
            "--context-len", "16",
            "--batch-size", "8",
            "--rtg-scale", "1000",
            "--init-from", "checkpoints_bc/bc_best.pt",
            "--out", "checkpoints_dt_norc",
        ],
    ),
]


def finished(out_name: str) -> bool:
    folder = os.path.join(EXP, out_name)
    log_path = os.path.join(folder, "loss_log.json")
    best_path = os.path.join(folder, "dt_best.pt")
    if not (os.path.exists(log_path) and os.path.exists(best_path)):
        return False
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    return bool(log) and log[-1].get("epoch") == EPOCHS and len(log) >= EPOCHS


def ensure_norc() -> None:
    index = os.path.join(EXP, "data", "graph_cache", "rtg_norc", "index.json")
    ready = False
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            meta = json.load(f)["meta"]
        ready = (
            meta.get("use_rq") is True and meta.get("use_protect") is True
            and meta.get("use_ri") is True and meta.get("use_rc") is False
        )
    if ready:
        print("rtg_norc 已存在且开关正确，跳过回报重组", flush=True)
        return
    print("生成 rtg_norc：去掉 R_c，保留原轨迹", flush=True)
    subprocess.run(
        [sys.executable, "reward_parts.py", "apply", "--set", "norc", "--no-rc"],
        check=True, cwd=DT,
    )


def preflight() -> None:
    bc = os.path.join(EXP, "checkpoints_bc", "bc_best.pt")
    norq = os.path.join(EXP, "data", "graph_cache", "rtg_norq", "index.json")
    split = os.path.join(EXP, "data", "graph_cache", "split_val.json")
    for path in (bc, norq, split):
        if not os.path.exists(path):
            raise SystemExit(f"缺少 {path}")
    with open(norq, encoding="utf-8") as f:
        trees = json.load(f)["trees"]
    with open(split, encoding="utf-8") as f:
        train = json.load(f)["train"]
    counts = {"expert": 0, "random": 0}
    for name in train:
        for row in trees.get(name, []):
            if row["policy"] in counts:
                counts[row["policy"]] += 1
    if counts["random"] == 0:
        raise SystemExit("norq 回报索引中没有随机轨迹")
    print(
        f"训练树 {len(train)} 棵：专家轨迹 {counts['expert']} 条，"
        f"随机轨迹 {counts['random']} 条",
        flush=True,
    )
    print("去园艺知识只使用上述随机轨迹，且不加载 BC。", flush=True)


def main():
    preflight()
    ensure_norc()
    for label, out_name, args in JOBS:
        if finished(out_name):
            print(f"[{label}] 已完成 {EPOCHS} epoch，跳过", flush=True)
            continue
        print("\n" + "=" * 72)
        print(f"[{label}] 开始训练 -> experiments/{out_name}")
        print("=" * 72, flush=True)
        subprocess.run([sys.executable, "train.py", *args], check=True, cwd=DT)
    print("\n两个变体的训练阶段结束。接下来运行：")
    print("py -3.12 experiments/scripts/run_e4_remaining_eval.py --workers 8")


if __name__ == "__main__":
    main()
