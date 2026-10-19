"""schema=2 轨迹转图缓存；不读取或写入 v1 canonical 数据目录。"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from config import CFG
from encoder import arrays_to_data, graph_arrays
from reward import KPI_KEYS, TreeContext
from tree_data import BRANCH
from v2_manifest import (V2_CACHE_ROOT, V2_TRAJ_ROOT, assert_v2_write_path,
                         load_manifest, manifest_hash)


SCHEMA_VERSION = 2


def _compact(arr: dict) -> dict:
    return {
        "x": arr["x"].astype(np.float16),
        "edge_index": arr["edge_index"].astype(np.int16),
        "edge_attr": arr["edge_attr"].astype(np.float16),
        "g": arr["g"].astype(np.float32),
        "root": int(arr["root"]),
    }


def convert_traj(tr: dict) -> dict:
    if tr.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{tr.get('tree')} 不是 schema=2 轨迹")
    sks, actions = tr["skeletons"], tr["actions"]
    pms = tr.get("parent_maps") or [None] * len(sks)
    n0 = len(np.unique(sks[0][:, BRANCH]))
    ctx = TreeContext(sks[0], CFG.reward, seed=tr["metadata"]["seed"])
    kpi_rows = np.asarray(tr["kpis"], dtype=np.float32)
    if kpi_rows.shape != (len(actions), len(KPI_KEYS)):
        raise ValueError(f"{tr['tree']} KPI shape={kpi_rows.shape}，动作数={len(actions)}")

    graphs, labels = [], []
    dk = np.zeros((len(actions), len(KPI_KEYS)), dtype=np.float32)
    dk_mask = np.zeros(len(actions), dtype=bool)
    components = np.zeros((len(actions), 3), dtype=np.float32)
    for t, action in enumerate(actions):
        sk = sks[t]
        m = len(np.unique(sk[:, BRANCH]))
        kp = {k: float(kpi_rows[t, i]) for i, k in enumerate(KPI_KEYS)}
        arr = graph_arrays(
            sk, pms[t], pruning_ratio=(n0 - m) / max(n0, 1),
            schema_version=SCHEMA_VERSION, tree_context=ctx, kpis=kp)
        if action >= 0:
            idx = int(np.searchsorted(arr["bids"], action))
            if idx >= len(arr["bids"]) or arr["bids"][idx] != action:
                raise ValueError(f"{tr['tree']} 第 {t} 步动作 {action} 不在当前图中")
            labels.append(idx)
            if t + 1 >= len(kpi_rows):
                raise ValueError(f"{tr['tree']} 第 {t} 步缺少剪后 KPI")
            dk[t] = kpi_rows[t + 1] - kpi_rows[t]
            dk_mask[t] = True
        else:
            labels.append(-1)
        graphs.append(_compact(arr))
        part = tr["reward_parts"][t]
        components[t] = [part.get("rq", 0.0), part.get("rc", 0.0), part.get("cost", 0.0)]

    md = tr["metadata"]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": tr["dataset_version"],
        "policy": tr["policy"],
        "variant": int(tr.get("variant", 0)),
        "total_reward": float(md["total_reward"]),
        "final_ratio": float(md["final_pruning_ratio"]),
        "stop_reason": md["stop_reason"],
        "graphs": graphs,
        "labels": np.asarray(labels, dtype=np.int32),
        "rtg": np.asarray(tr["returns_to_go"], dtype=np.float32),
        "rewards": np.asarray(tr["rewards"], dtype=np.float32),
        "dk": dk,
        "dk_mask": dk_mask,
        "components": components,  # [rq, rc, prune_cost]，均为未乘 alpha/gamma
    }


def _worker(name: str):
    try:
        with (V2_TRAJ_ROOT / f"{name}.pkl").open("rb") as f:
            trajs = pickle.load(f)
        return name, [convert_traj(t) for t in trajs], None
    except Exception as exc:
        return name, None, repr(exc)


def load_split() -> dict:
    manifest = load_manifest()
    return {"train": list(manifest["train"]), "val": list(manifest["val"])}


def load_cache(names: list[str], policies: tuple[str, ...] | None = None,
               reward_set: str = "full",
               cache_root: str | os.PathLike | None = None) -> list[dict]:
    root = Path(cache_root).resolve() if cache_root else V2_CACHE_ROOT
    out = []
    for name in names:
        path = root / f"{name}.pkl"
        if not path.exists():
            continue
        rtg_path = root / f"rtg_{reward_set}" / f"{name}.pkl"
        if not rtg_path.exists():
            raise FileNotFoundError(
                f"缺少 {rtg_path}；先运行 reward_parts_v2.py calibrate/apply --set {reward_set}")
        with path.open("rb") as f:
            trajs = pickle.load(f)
        with rtg_path.open("rb") as f:
            returns = pickle.load(f)
        if len(returns) != len(trajs):
            raise ValueError(f"{name} 的 cache 与 rtg_{reward_set} 轨迹数不一致")
        for tr, ret in zip(trajs, returns):
            if tr.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{path} schema 不一致")
            if tr["policy"] != ret["policy"] or len(tr["labels"]) != len(ret["rtg"]):
                raise ValueError(f"{name} 的 cache 与 rtg_{reward_set} 未对齐")
            if "rewards" not in ret or len(ret["rewards"]) != len(tr["labels"]):
                raise ValueError(f"{name} 的 rtg_{reward_set} 缺逐步 rewards；请重新 apply --overwrite")
            if policies is None or tr["policy"] in policies:
                item = dict(tr)
                item["rtg"] = ret["rtg"]
                item["rewards"] = ret["rewards"]
                item["total_reward"] = ret["total_reward"]
                item["tree"] = name
                out.append(item)
    return out


def cached_to_data(graph: dict):
    return arrays_to_data(graph)


def _write_meta(manifest: dict) -> None:
    meta = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": CFG.data.v2_version,
        "manifest_hash": manifest_hash(manifest),
        "branch_feature_dim": 30,
        "global_feature_dim": 8,
        "kpi_keys": list(KPI_KEYS),
    }
    path = assert_v2_write_path(V2_CACHE_ROOT / "meta.json")
    if path.exists():
        with path.open(encoding="utf-8") as f:
            old = json.load(f)
        if old != meta:
            raise SystemExit(f"已有 v2 cache meta 与当前配置不一致：{path}")
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "all"), default="all")
    ap.add_argument("--names", default=None)
    ap.add_argument("--max-trees", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 4, 1))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest()
    root = assert_v2_write_path(V2_CACHE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    _write_meta(manifest)
    split_path = assert_v2_write_path(root / "split_val.json")
    expected_split = load_split()
    if split_path.exists():
        with split_path.open(encoding="utf-8") as f:
            if json.load(f) != expected_split:
                raise SystemExit("已有 v2 split_val.json 与 manifest 不一致")
    else:
        with split_path.open("w", encoding="utf-8") as f:
            json.dump(expected_split, f, ensure_ascii=False, indent=2)

    names = expected_split["train"] if args.split == "train" else expected_split["val"]
    if args.split == "all":
        names = expected_split["train"] + expected_split["val"]
    if args.names:
        names = [x.strip() for x in args.names.split(",") if x.strip()]
    if args.max_trees is not None:
        names = names[:args.max_trees]
    missing = [n for n in names if not (V2_TRAJ_ROOT / f"{n}.pkl").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} 棵缺 v2 轨迹，示例：{missing[:5]}")
    todo = names if args.overwrite else [n for n in names if not (root / f"{n}.pkl").exists()]

    index_path = root / "index.json"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (name, trajs, err) in enumerate(pool.imap_unordered(_worker, todo), 1):
            if err:
                print(f"[{i}/{len(todo)}] {name} 失败：{err}", flush=True)
                continue
            path = assert_v2_write_path(root / f"{name}.pkl")
            with path.open("wb") as f:
                pickle.dump(trajs, f, protocol=4)
            index[name] = [{
                "policy": t["policy"], "variant": t["variant"],
                "steps": len(t["labels"]), "total_reward": t["total_reward"],
                "stop_reason": t["stop_reason"],
            } for t in trajs]
            print(f"[{i}/{len(todo)}] {name} 完成，累计 {(time.time()-t0)/60:.1f} min", flush=True)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"完成：{root}")


if __name__ == "__main__":
    main()
