"""将已生成的30棵上限轨迹精炼为20训练+10选模数据集。

原始 ``trajectories_v2`` 只读；输出到 ``trajectories_v2_curated``。
固定操作成本从候选净收益中扣除，不改变同一步候选动作排序，只重建自主
STOP 位置、回报和 RTG。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dt_pruning"))

from config import CFG  # noqa: E402
from v2_manifest import (MANIFEST_PATH, V2_TRAJ_ROOT, assert_v2_write_path,  # noqa: E402
                         config_fingerprint, manifest_hash)

RAW_ROOT = Path(CFG.data.work_root) / "data" / "trajectories_v2"
OLD_MANIFEST = Path(CFG.data.work_root) / "data" / "manifests" / "v2_manifest.json"


def ratio(sk, initial_branches: int) -> float:
    return (initial_branches - len(np.unique(sk[:, 5].astype(np.int64)))) / max(initial_branches, 1)


def stop_index(tr: dict, fixed_cost: float) -> int:
    """返回新 STOP 所在状态序号；其前面的动作保留。"""
    n0 = tr["metadata"]["initial_branch_count"]
    low = 0
    for t, old_net in enumerate(tr["metadata"]["best_net_history"]):
        if not np.isfinite(old_net):
            continue
        low = low + 1 if old_net - fixed_cost <= CFG.traj.v2_net_gain_threshold else 0
        state_ratio = ratio(tr["skeletons"][min(t, len(tr["skeletons"]) - 1)], n0)
        if state_ratio >= CFG.traj.v2_min_stop_ratio and low >= CFG.traj.v2_stop_patience:
            return min(t, len(tr["skeletons"]) - 1)
    return len(tr["skeletons"]) - 1


def fixed_cost_stop_index(tr: dict, fixed_cost: float, patience: int) -> int:
    n0 = tr["metadata"]["initial_branch_count"]
    low = 0
    for t, old_net in enumerate(tr["metadata"]["best_net_history"]):
        if not np.isfinite(old_net):
            continue
        low = low + 1 if old_net - fixed_cost <= 0 else 0
        state_ratio = ratio(tr["skeletons"][min(t, len(tr["skeletons"]) - 1)], n0)
        if state_ratio >= CFG.traj.v2_min_stop_ratio and low >= patience:
            return min(t, len(tr["skeletons"]) - 1)
    return len(tr["skeletons"]) - 1


def relative_stop_index(tr: dict, fraction: float) -> int:
    """边际净收益相对该轨迹历史峰值持续下降时停止。"""
    n0 = tr["metadata"]["initial_branch_count"]
    low, peak = 0, 0.0
    for t, net in enumerate(tr["metadata"]["best_net_history"]):
        if not np.isfinite(net):
            continue
        peak = max(peak, float(net))
        low = low + 1 if peak > 0 and net <= fraction * peak else 0
        state_ratio = ratio(tr["skeletons"][min(t, len(tr["skeletons"]) - 1)], n0)
        if state_ratio >= CFG.traj.v2_min_stop_ratio and low >= CFG.traj.v2_stop_patience:
            return min(t, len(tr["skeletons"]) - 1)
    return len(tr["skeletons"]) - 1


def curate(tr: dict, fixed_cost: float) -> dict:
    cut = stop_index(tr, fixed_cost)
    actual_rewards = [
        float(tr["rewards"][i] - fixed_cost) for i in range(cut)]
    rewards = actual_rewards + [0.0]
    actions = list(tr["actions"][:cut]) + [CFG.traj.stop_action]
    parts = []
    for item in tr["reward_parts"][:cut]:
        rec = dict(item)
        rec["cost"] = float(rec.get("cost", 0.0) + fixed_cost)
        parts.append(rec)
    parts.append({"rq": 0.0, "ri": 0.0, "rc": 0.0, "cost": 0.0})
    output = dict(tr)
    output.update(
        dataset_version=CFG.data.v2_version,
        skeletons=tr["skeletons"][:cut + 1],
        parent_maps=tr["parent_maps"][:cut + 1],
        actions=actions,
        rewards=rewards,
        reward_parts=parts,
        returns_to_go=np.cumsum(np.asarray(rewards)[::-1])[::-1].tolist(),
        kpis=tr["kpis"][:cut + 1],
    )
    md = dict(tr["metadata"])
    md.update(
        stop_reason="marginal_gain_curated" if cut < len(tr["skeletons"]) - 1
        else "safety_cap",
        best_net_history=[
            float(x - fixed_cost) if np.isfinite(x) else float(x)
            for x in tr["metadata"]["best_net_history"][:cut + 1]],
        final_pruning_ratio=ratio(tr["skeletons"][cut], md["initial_branch_count"]),
        steps=len(actions),
        total_reward=float(sum(rewards)),
        fixed_action_cost=fixed_cost,
        net_gain_threshold=CFG.traj.v2_net_gain_threshold,
        source_dataset="pruning-v2",
    )
    output["metadata"] = md
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not OLD_MANIFEST.exists():
        raise SystemExit(f"缺原 manifest：{OLD_MANIFEST}")
    with OLD_MANIFEST.open(encoding="utf-8") as f:
        old_manifest = json.load(f)
    available = sorted(p.stem for p in RAW_ROOT.glob("*.pkl"))
    expected = CFG.traj.v2_train_trees + CFG.traj.v2_model_val_trees
    if len(available) != expected:
        raise SystemExit(f"需要恰好 {expected} 棵原始轨迹，当前 {len(available)} 棵")
    rng = np.random.default_rng(CFG.data.split_seed)
    selected = np.asarray(available)[rng.permutation(len(available))].tolist()
    train = sorted(selected[:CFG.traj.v2_train_trees])
    val = sorted(selected[CFG.traj.v2_train_trees:])
    split_of = {name: "train" if name in train else "val" for name in available}

    summaries = {"train": [], "val": []}
    transformed = {}
    relative_grid = {x: {"train": [], "val": []} for x in (.15, .20, .25, .30, .35, .40)}
    patience_grid = {x: {"train": [], "val": []} for x in (2, 3, 4, 5)}
    for name in available:
        with (RAW_ROOT / f"{name}.pkl").open("rb") as f:
            trajs = pickle.load(f)
        curated = [curate(tr, CFG.reward.v2_fixed_action_cost) for tr in trajs]
        transformed[name] = curated
        summaries[split_of[name]].extend(
            tr["metadata"]["final_pruning_ratio"] for tr in curated)
        for fraction, by_split in relative_grid.items():
            for tr in trajs:
                cut = relative_stop_index(tr, fraction)
                by_split[split_of[name]].append(
                    ratio(tr["skeletons"][cut], tr["metadata"]["initial_branch_count"]))
        for patience, by_split in patience_grid.items():
            for tr in trajs:
                cut = fixed_cost_stop_index(tr, CFG.reward.v2_fixed_action_cost, patience)
                by_split[split_of[name]].append(
                    ratio(tr["skeletons"][cut], tr["metadata"]["initial_branch_count"]))
    report = {
        key: {
            "trees": len(train if key == "train" else val),
            "trajectories": len(values),
            "ratio_min": float(np.min(values)),
            "ratio_mean": float(np.mean(values)),
            "ratio_median": float(np.median(values)),
            "ratio_max": float(np.max(values)),
            "in_027_030": float(np.mean((np.asarray(values) >= .27)
                                       & (np.asarray(values) <= .30))),
        } for key, values in summaries.items()
    }
    relative_report = {
        str(fraction): {
            split: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "in_027_030": float(np.mean(
                    (np.asarray(values) >= .27) & (np.asarray(values) <= .30))),
            } for split, values in by_split.items()
        } for fraction, by_split in relative_grid.items()
    }
    patience_report = {
        str(patience): {
            split: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q10": float(np.quantile(values, .1)),
                "q90": float(np.quantile(values, .9)),
            } for split, values in by_split.items()
        } for patience, by_split in patience_grid.items()
    }
    print(json.dumps({"fixed_action_cost": CFG.reward.v2_fixed_action_cost,
                      "train": train, "val": val, "summary": report,
                      "relative_stop_grid": relative_report,
                      "fixed_cost_patience_grid": patience_report},
                     ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    output_root = assert_v2_write_path(V2_TRAJ_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, trajs in transformed.items():
        path = assert_v2_write_path(output_root / f"{name}.pkl")
        if path.exists() and not args.overwrite:
            raise SystemExit(f"精炼轨迹已存在：{path}；拒绝部分覆盖")
        with path.open("wb") as f:
            pickle.dump(trajs, f, protocol=4)
    manifest = {
        "dataset_version": CFG.data.v2_version,
        "schema_version": 2,
        "config_hash": config_fingerprint(),
        "selection_seed": CFG.data.split_seed,
        "source_manifest": str(OLD_MANIFEST),
        "source_trajectories": str(RAW_ROOT),
        "train_pool_size": len(available),
        "train": train,
        "val": val,
        "eval_val": old_manifest["val"],
        "test": {"sealed": True, "trees": []},
        "trajectory_counts": old_manifest["trajectory_counts"],
        "fixed_action_cost": CFG.reward.v2_fixed_action_cost,
        "paths": {
            "trajectories": str(V2_TRAJ_ROOT),
            "graph_cache": str(Path(CFG.data.work_root) / CFG.data.v2_cache_dir),
        },
        "curation_report": report,
    }
    path = assert_v2_write_path(MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"写入 {path}，manifest hash={manifest_hash(manifest)[:16]}")


if __name__ == "__main__":
    main()
