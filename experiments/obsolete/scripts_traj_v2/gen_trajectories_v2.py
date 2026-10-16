"""生成 v2 训练轨迹：N 棵树 × (1 条专家 + 1 条随机)。

与 dt_pruning/trajectories.py（v1）的三点区别：

1. 每棵树 1 条专家 + 1 条随机（v1 是 1+4），数据量从约 18GB 降到约 8GB。
2. 专家轨迹的目标剪枝比按树轮转分档（v1 全部剪到 30% 上限才停）。
   目的：让"何时停手"这件事在训练数据里有变化，不再只有一个答案。
3. 训练树只从全量 85/15 划分的 train 侧取（v1 按文件名顺序取，导致部分训练树
   同时落在 test 列表里）。本脚本保证与 225 棵全量测试树零重叠。

用法（工作目录下执行）：
    python experiments/scripts/gen_trajectories_v2.py --n-trees 100 --workers 12

--n-trees 是"本次再往下取多少棵"，不是"总共取前多少棵"：
第一次跑取 train 侧第 1-100 棵；再跑一次同样的命令，接着取第 101-200 棵，
已生成的不会重跑。取到哪一棵由输出目录里已有的 pkl 推断，
中途失败的树会在下一次运行时补上。

输出：experiments/data/trajectories_v2/
    {树名}.pkl    每棵树 2 条轨迹
    split.pkl     train/test 划分 + 每棵树的专家目标剪枝比
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dt_pruning"))

from config import CFG
from trajectories import rollout, split_trees
from tree_data import collect_tree_dirs, tree_name

# 专家目标剪枝比档位，按树轮转。
# 论文理想区间 [0.18, 0.32]，硬上限 0.30；0.16 略低于下界，提供"剪得偏少"的样本。
EXPERT_TARGETS = [0.16, 0.20, 0.23, 0.26, 0.30]


def stable_seed(name: str) -> int:
    """跨进程可复现的种子（v1 用 hash(name)，受 PYTHONHASHSEED 影响不可复现）。"""
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


def _worker(task):
    tree_dir, slot = task
    name = tree_name(tree_dir)
    target = EXPERT_TARGETS[slot % len(EXPERT_TARGETS)]
    try:
        base = stable_seed(name)
        expert = rollout(tree_dir, "expert", seed=base, target_ratio=target)
        rand_target = float(np.random.default_rng(base + 1000).uniform(
            CFG.traj.random_target_lo, CFG.traj.random_target_hi))
        rnd = rollout(tree_dir, "random", seed=base + 1000,
                      target_ratio=rand_target)
        return name, [expert, rnd], target, None
    except Exception as e:          # 单棵树失败不阻塞整批
        return name, None, target, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trees", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 4, 1))
    ap.add_argument("--out", type=str,
                    default=os.path.join("data", "trajectories_v2"))
    args = ap.parse_args()

    all_dirs = collect_tree_dirs(CFG.data.data_root)
    train_dirs, test_dirs = split_trees(all_dirs)      # 与 v1 同一种子

    out_root = os.path.join(CFG.data.work_root, args.out)
    os.makedirs(out_root, exist_ok=True)

    # 顺着 train 侧往下走：跳过已生成的，从第一个缺的开始再取 n_trees 棵
    have = {f[:-4] for f in os.listdir(out_root)
            if f.endswith(".pkl") and f != "split.pkl"}
    start = 0
    while start < len(train_dirs) and tree_name(train_dirs[start]) in have:
        start += 1
    sel = train_dirs[start: start + args.n_trees]

    print(f"全量 {len(all_dirs)} 棵；划分 train {len(train_dirs)} / test {len(test_dirs)}")
    print(f"已生成 {start} 棵，本次从 train 侧第 {start + 1} 棵起再取 {len(sel)} 棵"
          f"（与 225 棵 test 树无重叠）")
    print(f"专家目标剪枝比 {EXPERT_TARGETS}，每档约 {max(len(sel),1)//len(EXPERT_TARGETS)} 棵")
    print(f"输出目录 {out_root}", flush=True)

    os.makedirs(out_root, exist_ok=True)
    # 档位按 train 侧的全局位置轮转，保证扩量后各档仍然均匀
    tasks = [(d, start + i) for i, d in enumerate(sel)]
    meta, done, t0 = {}, 0, time.time()

    with Pool(args.workers) as pool:
        for name, trajs, target, err in pool.imap_unordered(_worker, tasks):
            done += 1
            if err:
                print(f"[{done}/{len(tasks)}] {name} 失败: {err}", flush=True)
                continue
            with open(os.path.join(out_root, f"{name}.pkl"), "wb") as f:
                pickle.dump(trajs, f, protocol=4)
            meta[name] = target
            if done % 10 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] 已耗时 {(time.time()-t0)/60:.1f} min",
                      flush=True)

    split_path = os.path.join(out_root, "split.pkl")
    old = {}
    if os.path.exists(split_path):
        with open(split_path, "rb") as f:
            old = pickle.load(f)
    names = sorted(set(old.get("train", [])) | set(meta))
    with open(split_path, "wb") as f:
        pickle.dump({
            "train": names,
            "test": [tree_name(d) for d in test_dirs],      # 全量 225 棵
            "expert_target": {**old.get("expert_target", {}), **meta},
            "expert_targets_grid": EXPERT_TARGETS,
        }, f)
    print(f"完成。本轮 {len(meta)} 棵，累计 {len(names)} 棵；划分文件 {split_path}")


if __name__ == "__main__":
    main()
