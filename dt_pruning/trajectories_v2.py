"""生成 schema=2 的结构化轨迹；与 v1 数据目录完全并列。

每棵树默认生成 1 条园艺专家、2 条 Top-k 扰动和 2 条 KPI 引导轨迹。
已存在文件默认跳过，只有显式 ``--overwrite`` 才允许重算 v2 文件。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from config import CFG
from kpi_scales_v2 import activate_v2_kpi_scales
from reward import (KPI_KEYS, TreeContext, branch_context, branch_quality_reward,
                    composite_reward_v2, compute_branch_props, compute_kpis,
                    continuous_constraint_penalty, detect_crossing, pruning_cost,
                    reward_for_set)
from trajectories import expert_scores
from tree_data import TreeState, collect_tree_dirs, load_skeleton, skeleton_path, tree_name
from v2_manifest import (MANIFEST_PATH, V2_TRAJ_ROOT, assert_v2_write_path,
                         load_manifest, manifest_hash)


POLICIES = ("gardening", "topk", "kpi", "random")


def stable_seed(name: str, policy: str, variant: int) -> int:
    raw = f"{CFG.traj.seed}:{name}:{policy}:{variant}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def _clone_state(state: TreeState) -> TreeState:
    return TreeState(state.skeleton.copy(), state.initial_nodes, state.initial_branch_count)


def _candidate_subset(legal: list[int], rng: np.random.Generator,
                      candidate_cap: int | None = None,
                      forced_bids: list[int] | None = None) -> list[int]:
    cap = candidate_cap or CFG.traj.v2_candidate_cap
    forced = [int(b) for b in (forced_bids or []) if b in legal][:cap]
    if len(legal) <= cap:
        return sorted(legal)
    remaining = [b for b in legal if b not in forced]
    sampled = rng.choice(
        remaining, size=cap - len(forced), replace=False).astype(int).tolist()
    return sorted(forced + sampled)


def evaluate_candidates(state: TreeState, props: dict, k_before: dict,
                        crossing: list, ctx: TreeContext,
                        rng: np.random.Generator,
                        candidate_cap: int | None = None,
                        forced_bids: list[int] | None = None) -> list[dict]:
    """固定预算的一步反事实评估；最多评估 32 根合法枝。"""
    legal = state.legal_actions(CFG.data.min_nodes_abs, CFG.data.min_nodes_ratio)
    scores = expert_scores(state, props, crossing)
    records = []
    for bid in _candidate_subset(legal, rng, candidate_cap, forced_bids):
        nxt = _clone_state(state)
        nxt.remove_branch(bid)
        if nxt.pruning_ratio() > CFG.traj.v2_max_pruning_ratio + 1e-9:
            continue
        props_after = compute_branch_props(nxt.skeleton, nxt.parent_map)
        k_after = compute_kpis(nxt.skeleton, props_after, CFG.reward, ctx)
        total, parts = composite_reward_v2(
            bid, state.skeleton, props, k_before, nxt.skeleton, props_after,
            k_after, crossing, CFG.reward)
        records.append({
            "bid": bid,
            "ratio_after": float(nxt.pruning_ratio()),
            "net": total,
            "rule_score": float(scores.get(bid, -np.inf)),
            "kpi_score": float(CFG.reward.beta * parts["ri"]
                               + CFG.reward.gamma * parts["rc"] - parts["cost"]),
            "parts": parts,
            "k_after": k_after,
        })
    return records


def _choose(records: list[dict], policy: str, variant: int,
            rng: np.random.Generator) -> dict | None:
    if not records:
        return None
    if policy == "gardening":
        return max(records, key=lambda x: (x["rule_score"], x["net"], -x["bid"]))
    if policy == "topk":
        ranked = sorted(records, key=lambda x: (x["rule_score"], x["net"]), reverse=True)
        k = min(CFG.traj.v2_topk, len(ranked))
        # 园艺轨迹已覆盖 rank-1；两条扰动优先使用 rank-2 / rank-3。
        return ranked[min(variant + 1, k - 1)]
    if policy == "kpi":
        ranked = sorted(records, key=lambda x: (x["kpi_score"], x["net"]), reverse=True)
        k = min(2, len(ranked))
        return ranked[min(variant, k - 1)]
    return records[int(rng.integers(len(records)))]


def rollout_v2(tree_dir: str, policy: str, variant: int,
               reward_set: str = "full") -> dict:
    activate_v2_kpi_scales()
    if policy not in POLICIES:
        raise ValueError(f"未知 v2 policy：{policy}")
    name = tree_name(tree_dir)
    seed = stable_seed(name, policy, variant)
    rng = np.random.default_rng(seed)
    sk0 = load_skeleton(skeleton_path(tree_dir))
    state = TreeState.from_skeleton(sk0)
    ctx = TreeContext(sk0, CFG.reward, seed=seed)

    skeletons = [state.skeleton.copy()]
    parent_maps = [dict(state.parent_map)]
    actions: list[int] = []
    rewards: list[float] = []
    reward_parts: list[dict] = []
    kpis: list[list[float]] = []
    best_net_history: list[float] = []
    low_gain_steps = 0
    stop_reason = "unknown"

    props = compute_branch_props(state.skeleton, state.parent_map)
    k_before = compute_kpis(state.skeleton, props, CFG.reward, ctx)
    crossing = detect_crossing(props, CFG.reward)

    while True:
        kpis.append([float(k_before[k]) for k in KPI_KEYS])
        records = evaluate_candidates(state, props, k_before, crossing, ctx, rng)
        if reward_set != "full":
            for record in records:
                record["net"] = reward_for_set(
                    record["parts"], reward_set, CFG.reward)
        best_net = max((r["net"] for r in records), default=float("-inf"))
        best_net_history.append(float(best_net))
        if best_net <= CFG.traj.v2_net_gain_threshold:
            low_gain_steps += 1
        else:
            low_gain_steps = 0

        ratio = state.pruning_ratio()
        if not records:
            stop_reason = "no_legal_action"
        elif ratio >= CFG.traj.v2_max_pruning_ratio:
            stop_reason = "safety_cap"
        elif ratio >= CFG.traj.v2_min_stop_ratio and low_gain_steps >= CFG.traj.v2_stop_patience:
            stop_reason = "marginal_gain"
        else:
            chosen = _choose(records, policy, variant, rng)
            if chosen is None:
                stop_reason = "no_candidate"
            else:
                action = int(chosen["bid"])
                state.remove_branch(action)
                actions.append(action)
                rewards.append(float(chosen["net"]))
                reward_parts.append(dict(chosen["parts"]))
                skeletons.append(state.skeleton.copy())
                parent_maps.append(dict(state.parent_map))
                props = compute_branch_props(state.skeleton, state.parent_map)
                k_before = compute_kpis(state.skeleton, props, CFG.reward, ctx)
                crossing = detect_crossing(props, CFG.reward)
                continue

        actions.append(CFG.traj.stop_action)
        # v2 STOP 标签由边际收益生成，不再用量级过小的 v1 分档奖励。
        rewards.append(0.0)
        reward_parts.append({"rq": 0.0, "ri": 0.0, "rc": 0.0, "cost": 0.0})
        break

    rtg = np.cumsum(np.asarray(rewards, dtype=np.float64)[::-1])[::-1]
    return {
        "tree": name,
        "policy": policy,
        "variant": variant,
        "dataset_version": CFG.data.v2_version,
        "schema_version": CFG.data.v2_schema_version,
        "skeletons": skeletons,
        "parent_maps": parent_maps,
        "actions": actions,
        "rewards": rewards,
        "reward_parts": reward_parts,
        "returns_to_go": rtg.tolist(),
        "kpis": kpis,
        "metadata": {
            "seed": seed,
            "stop_action": CFG.traj.stop_action,
            "stop_reason": stop_reason,
            "best_net_history": best_net_history,
            "candidate_cap": CFG.traj.v2_candidate_cap,
            "reward_set": reward_set,
            "initial_branch_count": state.initial_branch_count,
            "final_pruning_ratio": state.pruning_ratio(),
            "steps": len(actions),
            "total_reward": float(np.sum(rewards)),
            "manifest": str(MANIFEST_PATH),
        },
    }


def generate_tree(tree_dir: str) -> list[dict]:
    jobs = []
    for policy, count in (
        ("gardening", CFG.traj.v2_n_gardening),
        ("topk", CFG.traj.v2_n_topk),
        ("kpi", CFG.traj.v2_n_kpi),
        ("random", CFG.traj.v2_n_random),
    ):
        jobs.extend((policy, i) for i in range(count))
    return [rollout_v2(tree_dir, policy, variant) for policy, variant in jobs]


def _worker(tree_dir: str) -> tuple[str, list[dict] | None, str | None]:
    try:
        return tree_name(tree_dir), generate_tree(tree_dir), None
    except Exception as exc:
        return tree_name(tree_dir), None, repr(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "all"), default="all")
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-trees", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 4, 1))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest()
    names = manifest["train"] if args.split == "train" else manifest["val"]
    if args.split == "all":
        names = manifest["train"] + manifest["val"]
    if args.only:
        names = [n for n in names if n == args.only]
    if args.max_trees is not None:
        names = names[:args.max_trees]

    root = assert_v2_write_path(V2_TRAJ_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    all_dirs = {tree_name(d): d for d in collect_tree_dirs(CFG.data.data_root)}
    missing = [n for n in names if n not in all_dirs]
    if missing:
        raise SystemExit(f"原始树目录缺失：{missing}")
    todo = names if args.overwrite else [n for n in names if not (root / f"{n}.pkl").exists()]
    print(f"v2 manifest={manifest_hash(manifest)[:16]}，计划 {len(names)} 棵，待生成 {len(todo)} 棵")

    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (name, trajs, err) in enumerate(pool.imap_unordered(_worker, [all_dirs[n] for n in todo]), 1):
            if err:
                print(f"[{i}/{len(todo)}] {name} 失败：{err}", flush=True)
                continue
            path = assert_v2_write_path(root / f"{name}.pkl")
            if path.exists() and not args.overwrite:
                print(f"[{i}/{len(todo)}] {name} 已存在，跳过", flush=True)
                continue
            with path.open("wb") as f:
                pickle.dump(trajs, f, protocol=4)
            steps = sum(len(t["actions"]) for t in trajs)
            print(f"[{i}/{len(todo)}] {name}：{len(trajs)} 条/{steps} 步，"
                  f"累计 {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
