"""v2 数据名单与路径隔离。

v2 只能写入 ``trajectories_v2`` / ``graph_cache_v2``；v1 目录永远只读。
manifest 从既有开发划分中固定抽取 50 棵训练树，并原样保留 31 棵验证树。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from config import CFG


WORK_ROOT = Path(CFG.data.work_root).resolve()
V1_TRAJ_ROOT = (WORK_ROOT / "data" / "trajectories").resolve()
V1_CACHE_ROOT = (WORK_ROOT / "data" / "graph_cache").resolve()
V2_TRAJ_ROOT = (WORK_ROOT / CFG.data.v2_trajectory_dir).resolve()
V2_CACHE_ROOT = (WORK_ROOT / CFG.data.v2_cache_dir).resolve()
E4_NOKNOW_TRAJ_ROOT = (WORK_ROOT / "data" / "trajectories_v2_e4_noknowledge").resolve()
E4_NOKNOW_CACHE_ROOT = (WORK_ROOT / "data" / "graph_cache_v2_e4_noknowledge").resolve()
CANDIDATE_KPI_ROOT = (WORK_ROOT / "data" / "candidate_kpi_v2_curated").resolve()
MANIFEST_PATH = (WORK_ROOT / CFG.data.v2_manifest).resolve()
SOURCE_SPLIT = V1_CACHE_ROOT / "split_val.json"


def _canonical(path: str | os.PathLike) -> Path:
    return Path(path).resolve()


def assert_v2_write_path(path: str | os.PathLike) -> Path:
    """拒绝任何指向 v1 或 v2 根目录之外的数据写入。"""
    p = _canonical(path)
    if p == V1_TRAJ_ROOT or p == V1_CACHE_ROOT or V1_TRAJ_ROOT in p.parents or V1_CACHE_ROOT in p.parents:
        raise ValueError(f"拒绝写入 v1 数据目录：{p}")
    allowed = (
        V2_TRAJ_ROOT, V2_CACHE_ROOT, E4_NOKNOW_TRAJ_ROOT,
        E4_NOKNOW_CACHE_ROOT, CANDIDATE_KPI_ROOT, MANIFEST_PATH.parent)
    if not any(p == root or root in p.parents for root in allowed):
        raise ValueError(f"v2 输出必须位于并列 v2 目录或 manifests 目录：{p}")
    return p


def config_fingerprint() -> str:
    payload = {
        "dataset_version": CFG.data.v2_version,
        "schema_version": CFG.data.v2_schema_version,
        "seed": CFG.data.split_seed,
        "train_trees": CFG.traj.v2_train_trees,
        "model_val_trees": CFG.traj.v2_model_val_trees,
        "candidate_cap": CFG.traj.v2_candidate_cap,
        "fixed_action_cost": CFG.reward.v2_fixed_action_cost,
        "policies": {
            "gardening": CFG.traj.v2_n_gardening,
            "topk": CFG.traj.v2_n_topk,
            "kpi": CFG.traj.v2_n_kpi,
            "random": CFG.traj.v2_n_random,
        },
        "stop": {
            "min_ratio": CFG.traj.v2_min_stop_ratio,
            "max_ratio": CFG.traj.v2_max_pruning_ratio,
            "patience": CFG.traj.v2_stop_patience,
            "threshold": CFG.traj.v2_net_gain_threshold,
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def create_manifest(overwrite: bool = False) -> dict:
    path = assert_v2_write_path(MANIFEST_PATH)
    if path.exists() and not overwrite:
        return load_manifest(path)
    if not SOURCE_SPLIT.exists():
        raise FileNotFoundError(f"缺少既有开发划分：{SOURCE_SPLIT}")
    with SOURCE_SPLIT.open(encoding="utf-8") as f:
        split = json.load(f)
    train_pool = sorted(split["train"])
    n = CFG.traj.v2_train_trees + CFG.traj.v2_model_val_trees
    if len(train_pool) < n:
        raise ValueError(f"训练池只有 {len(train_pool)} 棵，无法抽取 {n} 棵")
    rng = np.random.default_rng(CFG.data.split_seed)
    selected = rng.choice(train_pool, size=n, replace=False).tolist()
    train = sorted(selected[:CFG.traj.v2_train_trees])
    val = sorted(selected[CFG.traj.v2_train_trees:])
    eval_val = sorted(split["val"])
    manifest = {
        "dataset_version": CFG.data.v2_version,
        "schema_version": CFG.data.v2_schema_version,
        "config_hash": config_fingerprint(),
        "selection_seed": CFG.data.split_seed,
        "source_split": str(SOURCE_SPLIT),
        "train_pool_size": len(train_pool),
        "train": train,
        "val": val,
        "eval_val": eval_val,
        "test": {"sealed": True, "trees": []},
        "trajectory_counts": {
            "gardening": CFG.traj.v2_n_gardening,
            "topk": CFG.traj.v2_n_topk,
            "kpi": CFG.traj.v2_n_kpi,
            "random": CFG.traj.v2_n_random,
        },
        "paths": {
            "trajectories": str(V2_TRAJ_ROOT),
            "graph_cache": str(V2_CACHE_ROOT),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def load_manifest(path: str | os.PathLike = MANIFEST_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"缺少 v2 manifest：{p}；先运行 v2_manifest.py create")
    with p.open(encoding="utf-8") as f:
        manifest = json.load(f)
    required = {"dataset_version", "schema_version", "config_hash", "train", "val", "paths"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"v2 manifest 缺字段：{sorted(missing)}")
    if manifest["schema_version"] != 2:
        raise ValueError(f"仅支持 schema=2，实际为 {manifest['schema_version']}")
    overlap = set(manifest["train"]) & set(manifest["val"])
    if overlap:
        raise ValueError(f"训练/验证名单重叠：{sorted(overlap)}")
    if len(manifest["train"]) != CFG.traj.v2_train_trees:
        raise ValueError("manifest 训练树数量与当前配置不一致")
    if len(manifest["val"]) != CFG.traj.v2_model_val_trees:
        raise ValueError("manifest 选模树数量与当前配置不一致")
    return manifest


def manifest_hash(manifest: dict) -> str:
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("create", "check"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    manifest = create_manifest(args.overwrite) if args.command == "create" else load_manifest()
    print(f"manifest: {MANIFEST_PATH}")
    print(f"train={len(manifest['train'])}, val={len(manifest['val'])}, hash={manifest_hash(manifest)[:16]}")
    print(f"trajectories: {V2_TRAJ_ROOT}")
    print(f"graph_cache: {V2_CACHE_ROOT}")


if __name__ == "__main__":
    main()
