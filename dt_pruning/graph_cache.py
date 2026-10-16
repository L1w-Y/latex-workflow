"""轨迹 → 分支级图缓存（训练直接读缓存，不再每步从原始骨架建图）。

原始轨迹每步存完整骨架（全量约 27GB）；缓存只存图数组（float16/int16）
与动作在当前分枝列表中的局部序号，体积约小一个数量级，可整体驻留内存。

缓存格式（每树一个 pkl，内含该树全部轨迹）：
    [{"policy", "total_reward", "target_ratio", "final_ratio",
      "graphs": [{"x","edge_index","edge_attr","g","root"}] × T,
      "labels": int32[T]（分枝局部序号；STOP=-1）,
      "rtg": float32[T]}, ...]
另写 index.json（每树每条轨迹的 policy/total_reward/steps）与
split_val.json（训练树再划出的验证树清单，固定种子，调参只用验证集）。

用法：
    python graph_cache.py --workers 12         # 全量转换
    python graph_cache.py --max-trees 3        # 冒烟测试
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from config import CFG
from encoder import arrays_to_data, graph_arrays
from tree_data import BRANCH

TRAJ_ROOT = os.path.join(CFG.data.work_root, "data", "trajectories")
CACHE_ROOT = os.path.join(CFG.data.work_root, "data", "graph_cache")
VAL_RATIO = 0.10
VAL_SEED = 42


# ==============================================================================
# 划分
# ==============================================================================

def available_train_names() -> list[str]:
    with open(os.path.join(TRAJ_ROOT, "split.pkl"), "rb") as f:
        names = pickle.load(f)["train"]
    return [n for n in names if os.path.exists(os.path.join(TRAJ_ROOT, f"{n}.pkl"))]


def make_val_split(names: list[str]) -> dict:
    rng = np.random.default_rng(VAL_SEED)
    idx = rng.permutation(len(names))
    n_val = max(int(round(len(names) * VAL_RATIO)), 1)
    val = sorted(names[i] for i in idx[:n_val])
    train = sorted(names[i] for i in idx[n_val:])
    return {"train": train, "val": val}


def load_val_split() -> dict:
    path = os.path.join(CACHE_ROOT, "split_val.json")
    if not os.path.exists(path):
        return make_val_split(available_train_names())
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# 转换
# ==============================================================================

def _compact(arr: dict) -> dict:
    return {"x": arr["x"].astype(np.float16),
            "edge_index": arr["edge_index"].astype(np.int16),
            "edge_attr": arr["edge_attr"].astype(np.float16),
            "g": arr["g"].astype(np.float32),
            "root": int(arr["root"])}


def convert_traj(tr: dict) -> dict:
    n0 = len(np.unique(tr["skeletons"][0][:, BRANCH]))
    pms = tr.get("parent_maps") or [None] * len(tr["skeletons"])
    graphs, labels = [], []
    for t, a in enumerate(tr["actions"]):
        sk = tr["skeletons"][t]
        m = len(np.unique(sk[:, BRANCH]))
        arr = graph_arrays(sk, pms[t], pruning_ratio=(n0 - m) / max(n0, 1))
        if a >= 0:
            i = int(np.searchsorted(arr["bids"], a))
            if i >= len(arr["bids"]) or arr["bids"][i] != a:
                raise ValueError(f"{tr['tree']} 第 {t} 步动作 {a} 不在当前分枝列表中")
            labels.append(i)
        else:
            labels.append(-1)
        graphs.append(_compact(arr))
    md = tr["metadata"]
    return {"policy": tr["policy"], "total_reward": float(md["total_reward"]),
            "target_ratio": md.get("target_pruning_ratio"),
            "final_ratio": float(md["final_pruning_ratio"]),
            "graphs": graphs, "labels": np.asarray(labels, dtype=np.int32),
            "rtg": np.asarray(tr["returns_to_go"], dtype=np.float32)}


def _worker(name: str):
    try:
        with open(os.path.join(TRAJ_ROOT, f"{name}.pkl"), "rb") as f:
            trajs = pickle.load(f)
        out = [convert_traj(tr) for tr in trajs]
        with open(os.path.join(CACHE_ROOT, f"{name}.pkl"), "wb") as f:
            pickle.dump(out, f, protocol=4)
        summary = [{"policy": c["policy"], "total_reward": c["total_reward"],
                    "steps": len(c["labels"])} for c in out]
        return name, summary, None
    except Exception as e:
        return name, None, repr(e)


# ==============================================================================
# 读取
# ==============================================================================

def load_cache(names: list[str], policies: tuple[str, ...] | None = None,
               reward_set: str = "full") -> list[dict]:
    """读取图缓存；rtg 与 total_reward 取自 rtg_<reward_set>/（reward_parts.py apply 生成）。

    缓存内的 rtg 是生成轨迹时的旧奖励，缺回报旁路文件时报错，避免误用。
    """
    rtg_dir = os.path.join(CACHE_ROOT, f"rtg_{reward_set}")
    out = []
    for n in names:
        p = os.path.join(CACHE_ROOT, f"{n}.pkl")
        if not os.path.exists(p):
            continue
        rp = os.path.join(rtg_dir, f"{n}.pkl")
        if not os.path.exists(rp):
            raise SystemExit(f"缺回报文件 {rp}：请先运行 reward_parts.py compute / apply --set {reward_set}")
        with open(p, "rb") as f:
            trajs = pickle.load(f)
        with open(rp, "rb") as f:
            rets = pickle.load(f)
        if len(rets) != len(trajs):
            raise SystemExit(f"{n}：回报文件轨迹数 {len(rets)} 与缓存 {len(trajs)} 不一致")
        for c, r in zip(trajs, rets):
            if r["policy"] != c["policy"] or len(r["rtg"]) != len(c["labels"]):
                raise SystemExit(f"{n}：回报文件与缓存轨迹不对齐")
            c["rtg"], c["total_reward"] = r["rtg"], r["total_reward"]
            if policies is None or c["policy"] in policies:
                c["tree"] = n
                out.append(c)
    return out


def cached_to_data(g: dict):
    return arrays_to_data(g)


def load_index(reward_set: str | None = None) -> dict:
    """reward_set 为 None 时读缓存自带索引（旧奖励，仅供步数等结构信息）；否则读该回报集的总回报。"""
    if reward_set is None:
        with open(os.path.join(CACHE_ROOT, "index.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    with open(os.path.join(CACHE_ROOT, f"rtg_{reward_set}", "index.json"), "r", encoding="utf-8") as f:
        return json.load(f)["trees"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 4, 1))
    ap.add_argument("--max-trees", type=int, default=None)
    ap.add_argument("--names", type=str, default=None, help="只转换指定树，逗号分隔")
    ap.add_argument("--overwrite", action="store_true", help="已存在的缓存也重新转换")
    args = ap.parse_args()

    os.makedirs(CACHE_ROOT, exist_ok=True)
    names = available_train_names()
    split = make_val_split(names)
    with open(os.path.join(CACHE_ROOT, "split_val.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    print(f"可用训练树 {len(names)}：训练 {len(split['train'])}，验证 {len(split['val'])}")

    todo = names[:args.max_trees] if args.max_trees else names
    if args.names:
        todo = args.names.split(",")
    if not args.overwrite:
        todo = [n for n in todo if not os.path.exists(os.path.join(CACHE_ROOT, f"{n}.pkl"))]

    idx_path = os.path.join(CACHE_ROOT, "index.json")
    index = load_index() if os.path.exists(idx_path) else {}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, (name, summary, err) in enumerate(pool.imap_unordered(_worker, todo), 1):
            if err:
                print(f"[{k}/{len(todo)}] {name} 失败: {err}", flush=True)
                continue
            index[name] = summary
            print(f"[{k}/{len(todo)}] {name} 完成，已耗时 {(time.time()-t0)/60:.1f} min",
                  flush=True)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f)
    print("完成，缓存目录:", CACHE_ROOT)


if __name__ == "__main__":
    main()
