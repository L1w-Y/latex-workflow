"""离线轨迹生成：专家策略 + 随机策略（论文 sec:model:dataset）。

轨迹格式（与论文一致，每步记录剪前状态）：
    {
      "tree": 树名, "policy": "expert" | "random",
      "skeletons": [np.ndarray(N_t, 6)] × (T+1),   # 含初始状态
      "actions":   [int] × T,                       # 被剪分枝ID；STOP 用 -1
      "rewards":   [float] × T,
      "returns_to_go": [float] × T,
      "metadata":  {...}
    }

用法：
    python trajectories.py --workers 8            # 全量生成（训练+测试树划分）
    python trajectories.py --only Lemon00_01      # 单棵调试
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from config import CFG
from reward import (LightContext, composite_reward, compute_branch_props,
                    compute_kpis, constraint_penalty, detect_crossing,
                    is_drooping, is_inward, is_medium_short, is_sturdy,
                    is_vigorous, is_weak, mean_length, mean_radius, stop_reward)
from tree_data import (TreeState, branch_ids, collect_tree_dirs, load_skeleton,
                       skeleton_path, tree_name)


# ==============================================================================
# 策略
# ==============================================================================

def expert_action(state: TreeState, props: dict, crossing: list,
                  light_ctx: LightContext) -> int | None:
    """启发式专家策略（论文 sec:model:dataset）。

    打分规则与枝条质量奖励同源，另加体积惩罚 -λ_exp·V_b；
    所有候选最高分 <= 0 时自动终止（返回 None）。
    """
    best_bid, best_score = None, 0.0
    for bid, score in expert_scores(state, props, crossing).items():
        if score > best_score:
            best_bid, best_score = bid, score
    return best_bid


def expert_scores(state: TreeState, props: dict, crossing: list) -> dict[int, float]:
    """专家对每个合法候选枝的打分（按合法动作顺序）。"""
    cfg = CFG.reward
    canopy_center = state.skeleton[:, :3].mean(axis=0)
    avg_len = mean_length(props, cfg.l_eff)
    avg_r = mean_radius(props, cfg.l_eff)
    cross_of = {}
    for ba, bb, sev, worse in crossing:
        cross_of[worse] = cfg.omega_cross_inferior * sev
        other = bb if worse == ba else ba
        cross_of.setdefault(other, cfg.omega_cross_other * sev)

    legal = state.legal_actions(CFG.data.min_nodes_abs, CFG.data.min_nodes_ratio)
    scores: dict[int, float] = {}
    for bid in legal:
        p = props.get(bid)
        if p is None:
            continue
        score = cross_of.get(bid, 0.0)
        if is_inward(p, canopy_center, cfg):
            score += cfg.omega_in
        if is_vigorous(p, avg_len, cfg):
            score += cfg.omega_vig
        if is_drooping(p, cfg):
            score += cfg.omega_drp
        if is_weak(p, avg_r, cfg):
            score += cfg.omega_wk
        if score == 0.0:
            if is_medium_short(p, avg_len, cfg):
                score -= cfg.pi_ms
            if is_sturdy(p, avg_r, cfg):
                score -= cfg.pi_st
        score -= CFG.traj.expert_volume_penalty * p.volume
        scores[bid] = score
    return scores


def random_action(state: TreeState, rng: np.random.Generator) -> int | None:
    legal = state.legal_actions(CFG.data.min_nodes_abs, CFG.data.min_nodes_ratio)
    if not legal:
        return None
    return int(rng.choice(legal))


# ==============================================================================
# 单条轨迹展开
# ==============================================================================

def rollout(tree_dir: str, policy: str, seed: int,
            target_ratio: float | None = None) -> dict:
    """从初始树状态展开一条轨迹，返回轨迹字典。"""
    sk0 = load_skeleton(skeleton_path(tree_dir))
    state = TreeState.from_skeleton(sk0)
    light_ctx = LightContext(sk0, CFG.reward, seed=seed)
    rng = np.random.default_rng(seed)
    max_ratio = CFG.traj.max_pruning_ratio

    skeletons = [state.skeleton.copy()]
    parent_maps = [dict(state.parent_map)]  # 每步父子关系预存，训练免重算
    actions, rewards = [], []

    # 状态缓存：上一步的"剪后"结果即下一步的"剪前"，避免重复计算
    props = compute_branch_props(state.skeleton, state.parent_map)
    k_before = compute_kpis(state.skeleton, props, CFG.reward, light_ctx)
    crossing = detect_crossing(props, CFG.reward)

    while True:
        if policy == "expert":
            action = expert_action(state, props, crossing, light_ctx)
        else:
            action = random_action(state, rng)

        # 终止判定
        stop = action is None
        if not stop and target_ratio is not None \
                and state.pruning_ratio() >= target_ratio:
            stop = True
        if not stop and state.pruning_ratio() >= max_ratio:
            stop = True

        if stop:
            r = stop_reward(state.pruning_ratio(), CFG.reward)
            actions.append(CFG.traj.stop_action)
            rewards.append(r)
            break

        sk_before = state.skeleton
        props_before = props
        state.remove_branch(action)
        props = compute_branch_props(state.skeleton, state.parent_map)
        k_after = compute_kpis(state.skeleton, props, CFG.reward, light_ctx)
        r, _ = composite_reward(action, sk_before, props_before, k_before,
                                state.skeleton, props, k_after,
                                crossing, CFG.reward)
        actions.append(action)
        rewards.append(r)
        skeletons.append(state.skeleton.copy())
        parent_maps.append(dict(state.parent_map))
        # 更新缓存供下一步使用
        crossing = detect_crossing(props, CFG.reward)
        k_before = k_after

    rtg = np.cumsum(rewards[::-1])[::-1].tolist()
    return {
        "tree": tree_name(tree_dir),
        "policy": policy,
        "skeletons": skeletons,
        "parent_maps": parent_maps,
        "actions": actions,
        "rewards": rewards,
        "returns_to_go": rtg,
        "metadata": {
            "stop_action": CFG.traj.stop_action,
            "target_pruning_ratio": target_ratio,
            "max_pruning_ratio": max_ratio,
            "final_pruning_ratio": state.pruning_ratio(),
            "initial_branch_count": state.initial_branch_count,
            "steps": len(actions),
            "total_reward": float(np.sum(rewards)),
            "seed": seed,
        },
    }


# ==============================================================================
# 批量生成
# ==============================================================================

def split_trees(tree_dirs: list[str]) -> tuple[list[str], list[str]]:
    """树级 85/15 划分（固定种子，全平台共用此划分）。"""
    rng = np.random.default_rng(CFG.data.split_seed)
    idx = rng.permutation(len(tree_dirs))
    n_train = int(round(len(tree_dirs) * CFG.data.train_ratio))
    train = [tree_dirs[i] for i in sorted(idx[:n_train])]
    test = [tree_dirs[i] for i in sorted(idx[n_train:])]
    return train, test


def _gen_one_tree(tree_dir: str, tree_idx: int = 0) -> list[dict]:
    name = tree_name(tree_dir)
    out = []
    base = abs(hash(name)) % (2 ** 31)
    tiers = CFG.traj.expert_target_tiers
    for i in range(CFG.traj.n_expert):
        # 专家目标剪枝比按树轮转分档（解决 STOP 监督单调）
        target = float(tiers[(tree_idx + i) % len(tiers)]) if tiers else None
        out.append(rollout(tree_dir, "expert", seed=base + i,
                           target_ratio=target))
    for j in range(CFG.traj.n_random):
        target = float(np.random.default_rng(base + 1000 + j).uniform(
            CFG.traj.random_target_lo, CFG.traj.random_target_hi))
        out.append(rollout(tree_dir, "random", seed=base + 1000 + j,
                           target_ratio=target))
    return out


def _worker(args: tuple[int, str]) -> tuple[str, list[dict] | None, str | None]:
    idx, tree_dir = args
    try:
        return tree_name(tree_dir), _gen_one_tree(tree_dir, idx), None
    except Exception as e:  # 单树失败不阻塞全量
        return tree_name(tree_dir), None, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 4, 1))
    ap.add_argument("--only", type=str, default=None, help="只生成指定树，如 Lemon00_01")
    ap.add_argument("--max-trees", type=int, default=None,
                    help="只生成前 N 棵树（按名称排序）")
    ap.add_argument("--out", type=str, default=os.path.join("data", "trajectories"))
    args = ap.parse_args()

    tree_dirs = collect_tree_dirs(CFG.data.data_root)
    if args.only:
        tree_dirs = [d for d in tree_dirs if tree_name(d) == args.only]
    if args.max_trees:
        tree_dirs = tree_dirs[:args.max_trees]
    train_dirs, test_dirs = split_trees(tree_dirs)
    print(f"总树数 {len(tree_dirs)}，训练 {len(train_dirs)}，测试 {len(test_dirs)}")

    out_root = os.path.join(CFG.data.work_root, args.out)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "split.pkl"), "wb") as f:
        pickle.dump({"train": [tree_name(d) for d in train_dirs],
                     "test": [tree_name(d) for d in test_dirs]}, f)

    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for name, trajs, err in pool.imap_unordered(
                _worker, list(enumerate(tree_dirs))):
            done += 1
            if err:
                print(f"[{done}/{len(tree_dirs)}] {name} 失败: {err}")
                continue
            with open(os.path.join(out_root, f"{name}.pkl"), "wb") as f:
                pickle.dump(trajs, f, protocol=4)
            el = time.time() - t0
            steps = sum(t["metadata"]["steps"] for t in trajs)
            print(f"[{done}/{len(tree_dirs)}] {name} 完成（{len(trajs)} 条，"
                  f"共 {steps} 步）已耗时 {el/60:.1f} min", flush=True)
    print("全部完成，输出目录:", out_root)


if __name__ == "__main__":
    main()
