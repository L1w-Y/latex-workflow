"""生成 E4“去园艺知识”的独立随机轨迹与缓存。

只写入 trajectories_v2_e4_noknowledge / graph_cache_v2_e4_noknowledge，
不修改完整模型使用的 v2 轨迹和缓存。动作均匀随机，STOP 与 RTG 使用
去掉 R_q 的 norq 净收益。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dt_pruning"))

from config import CFG  # noqa: E402
from graph_cache_v2 import convert_traj  # noqa: E402
from trajectories_v2 import rollout_v2  # noqa: E402
from tree_data import collect_tree_dirs, tree_name  # noqa: E402
from v2_manifest import (E4_NOKNOW_CACHE_ROOT, E4_NOKNOW_TRAJ_ROOT,  # noqa: E402
                         assert_v2_write_path, load_manifest, manifest_hash)


def _worker(job):
    name, tree_dir, count = job
    trajectories = [
        rollout_v2(tree_dir, "random", variant, reward_set="norq")
        for variant in range(count)
    ]
    return name, trajectories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-per-tree", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-trees", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.random_per_tree < 1:
        raise SystemExit("--random-per-tree 必须至少为1")

    manifest = load_manifest()
    names = manifest["train"] + manifest["val"]
    if args.max_trees:
        names = names[:args.max_trees]
    dirs = {tree_name(d): d for d in collect_tree_dirs(CFG.data.data_root)}
    missing = [n for n in names if n not in dirs]
    if missing:
        raise SystemExit(f"缺少原始树：{missing[:5]}")

    traj_root = assert_v2_write_path(E4_NOKNOW_TRAJ_ROOT)
    cache_root = assert_v2_write_path(E4_NOKNOW_CACHE_ROOT)
    rtg_root = assert_v2_write_path(cache_root / "rtg_norq")
    for root in (traj_root, cache_root, rtg_root):
        root.mkdir(parents=True, exist_ok=True)
    todo = names if args.overwrite else [
        n for n in names
        if not (traj_root / f"{n}.pkl").exists()
        or not (cache_root / f"{n}.pkl").exists()
        or not (rtg_root / f"{n}.pkl").exists()]
    print(
        f"去园艺知识数据：{len(names)}棵×{args.random_per_tree}条，"
        f"待生成{len(todo)}棵；输出不覆盖完整v2数据", flush=True)
    if args.dry_run:
        print(f"轨迹目录：{traj_root}\n缓存目录：{cache_root}")
        return

    index, started = {}, time.time()
    jobs = [(n, dirs[n], args.random_per_tree) for n in todo]
    pool = Pool(args.workers)
    try:
        for i, (name, trajectories) in enumerate(
                pool.imap_unordered(_worker, jobs), 1):
            converted = [convert_traj(t) for t in trajectories]
            returns = [{
                "policy": "random", "variant": t["variant"],
                "rewards": c["rewards"], "rtg": c["rtg"],
                "total_reward": c["total_reward"],
            } for t, c in zip(trajectories, converted)]
            with (traj_root / f"{name}.pkl").open("wb") as f:
                pickle.dump(trajectories, f, protocol=4)
            with (cache_root / f"{name}.pkl").open("wb") as f:
                pickle.dump(converted, f, protocol=4)
            with (rtg_root / f"{name}.pkl").open("wb") as f:
                pickle.dump(returns, f, protocol=4)
            index[name] = [{
                "policy": "random", "variant": c["variant"],
                "steps": len(c["labels"]),
                "total_reward": c["total_reward"],
                "stop_reason": c["stop_reason"],
            } for c in converted]
            print(
                f"[{i}/{len(todo)}] {name} 完成，"
                f"累计{(time.time() - started) / 60:.1f}min", flush=True)
    except KeyboardInterrupt:
        print("\n收到中断：正在强制结束全部worker；已落盘的整棵树保留。", flush=True)
        pool.terminate()
        pool.join()
        return
    else:
        pool.close()
        pool.join()

    # 重跑时也重建完整索引，避免只记录本轮新增树。
    all_index = {}
    for name in names:
        path = cache_root / f"{name}.pkl"
        if not path.exists():
            continue
        with path.open("rb") as f:
            rows = pickle.load(f)
        all_index[name] = [{
            "policy": x["policy"], "variant": x["variant"],
            "steps": len(x["labels"]), "total_reward": x["total_reward"],
            "stop_reason": x["stop_reason"],
        } for x in rows]
    meta = {
        "schema_version": 2, "manifest_hash": manifest_hash(manifest),
        "ablation": "no_gardening", "policies": ["random"],
        "reward_set": "norq", "random_per_tree": args.random_per_tree,
        "use_rule_features": False,
    }
    with (cache_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump(all_index, f, ensure_ascii=False)
    with (rtg_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "trees": all_index}, f, ensure_ascii=False, indent=2)
    with (cache_root / "split_val.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"train": manifest["train"], "val": manifest["val"]},
            f, ensure_ascii=False, indent=2)
    with (cache_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"完成：{cache_root}", flush=True)


if __name__ == "__main__":
    main()
