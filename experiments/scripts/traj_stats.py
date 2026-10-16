"""轨迹质量检查（风险试水·第一层，无需训练）。

对 experiments/data/trajectories/ 下已生成的轨迹做五项检查：
1. 专家 vs 随机回报分离度（一票否决项：分不开说明奖励信号无指导性）
2. 专家动作构成（缺陷枝类别占比；抽样树逐条回放统计）
3. 停止行为分布（专家最终剪枝比是否全部顶到上限，STOP 监督是否单调）
4. 奖励分量范围
5. 步数与 RTG 分布（决定训练窗口长度与 RTG 缩放）

用法（在 dt_pruning 目录外也可运行，自动处理路径）：
    python experiments/scripts/traj_stats.py
    python experiments/scripts/traj_stats.py --classify-trees 5 --max-files 30
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

# 让脚本可独立运行：把 dt_pruning 加入 import 路径
_DT_PRUNING = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "dt_pruning")
sys.path.insert(0, os.path.abspath(_DT_PRUNING))

from config import CFG  # noqa: E402

TRAJ_ROOT = os.path.join(CFG.data.work_root, "data", "trajectories")
RESULTS_DIR = os.path.join(CFG.data.work_root, "results")


def load_all(max_files: int | None, keep_full_expert_trees: int = 0
             ) -> tuple[list[dict], list[dict]]:
    files = sorted(f for f in os.listdir(TRAJ_ROOT)
                   if f.endswith(".pkl") and f != "split.pkl")
    if max_files:
        files = files[:max_files]
    # 逐棵读取、只保留轻量字段（单棵轨迹文件可达 200MB，全量载入会撑爆内存）；
    # 完整轨迹（含骨架）只为动作构成统计保留前 keep_full_expert_trees 棵专家轨迹
    trajs: list[dict] = []       # 轻量记录
    full_trajs: list[dict] = []  # 完整记录（少量）
    kept_expert_trees: set[str] = set()
    for i, fn in enumerate(files, 1):
        with open(os.path.join(TRAJ_ROOT, fn), "rb") as f:
            tree_trajs = pickle.load(f)
        for tr in tree_trajs:
            trajs.append({
                "tree": tr["tree"], "policy": tr["policy"],
                "actions": tr["actions"], "rewards": tr["rewards"],
                "returns_to_go": tr["returns_to_go"],
                "metadata": tr["metadata"],
            })
            if tr["policy"] == "expert" \
                    and len(kept_expert_trees) < keep_full_expert_trees:
                kept_expert_trees.add(tr["tree"])
                full_trajs.append(tr)
        del tree_trajs
        if i % 50 == 0 or i == len(files):
            print(f"  读取进度 {i}/{len(files)}", flush=True)
    return trajs, full_trajs


def check_separation(trajs: list[dict]) -> dict:
    """检查1：专家 vs 随机回报分离度（按树配对）。"""
    by_tree: dict[str, dict[str, list[float]]] = {}
    for tr in trajs:
        by_tree.setdefault(tr["tree"], {}).setdefault(tr["policy"], []).append(
            tr["metadata"]["total_reward"])
    diffs, expert_all, random_all = [], [], []
    for tree, d in by_tree.items():
        if "expert" in d and "random" in d:
            e, r = float(np.mean(d["expert"])), float(np.mean(d["random"]))
            diffs.append(e - r)
            expert_all.append(e)
            random_all.append(r)
    diffs = np.array(diffs)
    return {
        "n_trees_paired": len(diffs),
        "expert_mean": float(np.mean(expert_all)) if expert_all else None,
        "random_mean": float(np.mean(random_all)) if random_all else None,
        "diff_mean": float(diffs.mean()) if len(diffs) else None,
        "expert_better_rate": float((diffs > 0).mean()) if len(diffs) else None,
    }


def check_stop(trajs: list[dict]) -> dict:
    """检查3：停止行为分布。"""
    out = {}
    for policy in ("expert", "random"):
        ratios = [tr["metadata"]["final_pruning_ratio"]
                  for tr in trajs if tr["policy"] == policy]
        if not ratios:
            continue
        ratios = np.array(ratios)
        out[policy] = {
            "mean": float(ratios.mean()),
            "at_cap_rate": float((ratios >= CFG.traj.max_pruning_ratio - 1e-6).mean()),
            "in_ideal_range_rate": float(
                ((ratios >= CFG.reward.eta_lo) & (ratios <= CFG.reward.eta_hi)).mean()),
        }
    return out


def check_rewards(trajs: list[dict]) -> dict:
    """检查4：奖励范围。"""
    rs = np.array([r for tr in trajs for r in tr["rewards"]])
    return {
        "mean": float(rs.mean()), "std": float(rs.std()),
        "min": float(rs.min()), "max": float(rs.max()),
        "negative_rate": float((rs < 0).mean()),
    }


def check_steps_rtg(trajs: list[dict]) -> dict:
    """检查5：步数与 RTG 分布。"""
    steps = np.array([len(tr["actions"]) for tr in trajs])
    rtg0 = np.array([tr["returns_to_go"][0] for tr in trajs])
    return {
        "steps_mean": float(steps.mean()),
        "steps_p50": float(np.percentile(steps, 50)),
        "steps_p95": float(np.percentile(steps, 95)),
        "steps_max": int(steps.max()),
        "rtg0_mean": float(rtg0.mean()),
        "rtg0_p90": float(np.percentile(rtg0, 90)),
        "rtg0_max": float(rtg0.max()),
    }


def classify_expert_actions(trajs: list[dict], n_trees: int) -> dict:
    """检查2：专家动作构成（抽样树回放逐步分类）。"""
    from reward import (detect_crossing, is_drooping, is_inward, is_vigorous,
                        is_weak, is_medium_short, is_sturdy, mean_length,
                        mean_radius, compute_branch_props)
    from tree_data import build_parent_map

    counts = {"crossing": 0, "inward": 0, "vigorous": 0, "drooping": 0,
              "weak": 0, "protected_violation": 0, "other": 0, "total": 0}
    done = 0
    for tr in trajs:
        if tr["policy"] != "expert" or done >= n_trees:
            continue
        done += 1
        for sk, a in zip(tr["skeletons"], tr["actions"]):
            if a < 0:
                continue
            parent_map = build_parent_map(sk)
            props = compute_branch_props(sk, parent_map)
            p = props.get(a)
            if p is None:
                continue
            counts["total"] += 1
            crossing = detect_crossing(props, CFG.reward)
            in_pair = any(a in (ba, bb) for ba, bb, _, _ in crossing)
            center = sk[:, :3].mean(axis=0)
            avg_len, avg_r = mean_length(props, CFG.reward.l_eff), \
                mean_radius(props, CFG.reward.l_eff)
            defect = False
            if in_pair:
                counts["crossing"] += 1; defect = True
            if is_inward(p, center, CFG.reward):
                counts["inward"] += 1; defect = True
            if is_vigorous(p, avg_len, CFG.reward):
                counts["vigorous"] += 1; defect = True
            if is_drooping(p, CFG.reward):
                counts["drooping"] += 1; defect = True
            if is_weak(p, avg_r, CFG.reward):
                counts["weak"] += 1; defect = True
            if not defect:
                if is_medium_short(p, avg_len, CFG.reward) or \
                        is_sturdy(p, avg_r, CFG.reward):
                    counts["protected_violation"] += 1
                else:
                    counts["other"] += 1
    total = max(counts["total"], 1)
    return {"n_trees": done,
            **{k: round(v / total, 4) for k, v in counts.items() if k != "total"},
            "total_actions": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify-trees", type=int, default=3,
                    help="动作构成统计的抽样树数（0 跳过；逐树回放较耗时）")
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    trajs, full_trajs = load_all(args.max_files, args.classify_trees)
    n_trees = len({tr["tree"] for tr in trajs})
    print(f"已加载 {len(trajs)} 条轨迹，来自 {n_trees} 棵树\n")

    report = {"n_trajectories": len(trajs), "n_trees": n_trees,
              "separation": check_separation(trajs),
              "stop_behavior": check_stop(trajs),
              "reward_range": check_rewards(trajs),
              "steps_rtg": check_steps_rtg(trajs)}
    if args.classify_trees > 0:
        print("正在回放专家轨迹统计动作构成……")
        report["expert_action_composition"] = classify_expert_actions(
            full_trajs, args.classify_trees)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    out_dir = os.path.join(RESULTS_DIR, "paper", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "traj_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n已保存:", out)

    # 放行判断提示（最终判断由人做）
    sep = report["separation"]
    if sep.get("expert_better_rate") is not None:
        print("\n--- 放行参考 ---")
        print("专家优于随机的树占比: %.1f%%" % (sep["expert_better_rate"] * 100))
        stop = report["stop_behavior"].get("expert", {})
        if stop:
            print("专家顶到剪枝上限占比: %.1f%%（过高说明 STOP 监督单调）"
                  % (stop["at_cap_rate"] * 100))


if __name__ == "__main__":
    main()
