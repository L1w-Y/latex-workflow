"""把 trajectories_v2 里已生成的轨迹搬进 data/trajectories。

背景：轨迹文件名就是数据集树名（Lemon00_01.pkl），新旧两批同名，
直接拷进去会覆盖旧文件。这里给新文件统一加 _v2 后缀，
并同步改写 split.pkl 里的名字——train.py 是按 split.pkl 的名字去找文件的，
两边必须一致，否则会静默漏加载。

用法：
    python experiments/scripts/migrate_v2.py            # 演练，只打印将做什么
    python experiments/scripts/migrate_v2.py --apply    # 真正执行
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "dt_pruning"))
sys.path.insert(0, _HERE)

from config import CFG
from gen_trajectories_v2 import EXPERT_TARGETS
from trajectories import split_trees
from tree_data import collect_tree_dirs, tree_name

SUFFIX = "_v2"


def load_source_split(src: str, names: list[str]) -> dict:
    """源目录的 split.pkl；若生成被中断导致缺失，按 train 侧位置推断。"""
    path = os.path.join(src, "split.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    print("源目录缺 split.pkl（生成被中断），按 train 侧位置重建")
    train_dirs, test_dirs = split_trees(collect_tree_dirs(CFG.data.data_root))
    pos = {tree_name(d): i for i, d in enumerate(train_dirs)}
    got = set(names)
    train = [tree_name(d) for d in train_dirs if tree_name(d) in got]
    return {
        "train": train,
        "test": [tree_name(d) for d in test_dirs],
        "expert_target": {n: EXPERT_TARGETS[pos[n] % len(EXPERT_TARGETS)]
                          for n in train},
        "expert_targets_grid": EXPERT_TARGETS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join("data", "trajectories_v2"))
    ap.add_argument("--dst", default=os.path.join("data", "trajectories"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    src = os.path.join(CFG.data.work_root, args.src)
    dst = os.path.join(CFG.data.work_root, args.dst)
    os.makedirs(dst, exist_ok=True)

    files = sorted(f for f in os.listdir(src)
                   if f.endswith(".pkl") and f != "split.pkl")
    names = [f[:-4] for f in files]
    sp = load_source_split(src, names)

    print(f"源 {src}: {len(names)} 棵树")
    print(f"目标 {dst}")
    print(f"改名规则: {{树名}}.pkl -> {{树名}}{SUFFIX}.pkl")
    print(f"样例: {names[:3]} -> {[n + SUFFIX for n in names[:3]]}")

    clash = [n for n in names if os.path.exists(os.path.join(dst, n + ".pkl"))]
    print(f"其中 {len(clash)} 棵与目标目录同名（加后缀后不会覆盖）")

    if not args.apply:
        print("\n（演练模式，未改动任何文件。加 --apply 执行）")
        return

    moved = 0
    for n in names:
        s = os.path.join(src, n + ".pkl")
        d = os.path.join(dst, n + SUFFIX + ".pkl")
        if os.path.exists(d):
            print(f"  跳过，目标已存在: {n}{SUFFIX}")
            continue
        shutil.move(s, d)
        moved += 1
    print(f"已移动 {moved} 个文件")

    # 合并 split.pkl：名字带后缀，与移动后的文件一致
    dst_split = os.path.join(dst, "split.pkl")
    old = {}
    if os.path.exists(dst_split):
        with open(dst_split, "rb") as f:
            old = pickle.load(f)
    new_train = [n + SUFFIX for n in sp["train"]]
    merged = {
        "train": sorted(set(old.get("train", [])) | set(new_train)),
        "test": sp.get("test", old.get("test", [])),
        "expert_target": {**old.get("expert_target", {}),
                          **{n + SUFFIX: v for n, v in sp["expert_target"].items()}},
        "expert_targets_grid": sp.get("expert_targets_grid", EXPERT_TARGETS),
    }
    with open(dst_split, "wb") as f:
        pickle.dump(merged, f)
    print(f"合并后 split.pkl: train {len(merged['train'])} "
          f"(旧 {len(old.get('train', []))} + 新 {len(new_train)})，"
          f"test {len(merged['test'])}")

    left = [f for f in os.listdir(src) if f.endswith(".pkl") and f != "split.pkl"]
    print(f"源目录剩余 {len(left)} 个轨迹文件")


if __name__ == "__main__":
    main()
