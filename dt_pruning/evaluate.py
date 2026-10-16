"""统一评估协议：DT / BC rollout 与基线对比，逐树结果落盘 + 显著性检验。

纪律：
- 调参只用验证树（--split val），测试树（--split test）只用于最终报告
- 所有策略跑同一批树、同一随机种子；逐树明细落盘供配对检验
- 每步剪枝后状态立即更新，编码器在线编码当前真实状态
- 奖励/KPI 与训练完全同源（reward.py 唯一实现）

输出目录：experiments/results/<stage>/<exp>/<split>/
- stage=dev（默认）：调参与过程数据，exp 默认 adhoc
- stage=paper：写进论文的正式数据，必须指定 --exp（实验编号，如 E2_teacher），
  已有同名结果时拒绝覆盖（除非 --overwrite）；测试树只能以 paper 阶段运行

用法：
    python evaluate.py --policy dt --checkpoint checkpoints_dt/dt_best.pt --split val --exp tune_target
    python evaluate.py --policy greedy --split val --workers 8 --exp tune_target
    python evaluate.py --compare --split val --exp tune_target --ref dt
    python evaluate.py --policy dt --checkpoint ... --split test --stage paper --exp main
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
from torch_geometric.data import Batch

from config import CFG
from baselines.policies import greedy_action, tree_horizontal_radius
from encoder import arrays_to_data, graph_arrays
from reward import (LightContext, branch_context, branch_tags, composite_reward,
                    compute_branch_props, compute_kpis, detect_crossing,
                    reward_for_set, stop_reward)
from trajectories import expert_action
from tree_data import (TreeState, collect_tree_dirs, descendants,
                       load_skeleton, skeleton_path, tree_name)

RESULTS_DIR = os.path.join(CFG.data.work_root, "results")
OFFLINE_ALGOS = ("cql", "bcq", "iql")
LEARNED = ("dt", "bc", *OFFLINE_ALGOS)
KPI_KEYS = ["hierarchy", "crowding", "balance", "light", "potential", "stability"]


# ==============================================================================
# 模型加载
# ==============================================================================

def load_learned(checkpoint: str, device: torch.device) -> dict:
    path = checkpoint if os.path.isabs(checkpoint) \
        else os.path.join(CFG.data.work_root, checkpoint)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    kind = ckpt.get("kind")
    meta = ckpt["meta"]
    if kind == "bc":
        from train_bc import build_models
        encoder, model = build_models(device)
    elif kind == "dt":
        from train import build_models
        encoder, model = build_models(device, meta.get("max_timestep", 1024))
    elif kind in OFFLINE_ALGOS:
        from train_offline import build_models
        encoder, model = build_models(device, kind, meta["hparams"].get("bcq_threshold", 0.3))
    else:
        raise ValueError("旧版 checkpoint（全局分枝编号动作头）已不兼容，请用新脚本重新训练")
    encoder.load_state_dict(ckpt["encoder"])
    model.load_state_dict(ckpt["model"])
    encoder.eval(); model.eval()
    return {"kind": kind, "encoder": encoder, "model": model, "meta": meta,
            "device": device}


# ==============================================================================
# rollout
# ==============================================================================

class _Env:
    """评估环境：状态、KPI 与交叉检测缓存（剪后结果即下一步剪前）。"""

    def __init__(self, tree_dir: str, seed: int):
        sk0 = load_skeleton(skeleton_path(tree_dir))
        self.tree_dir = tree_dir
        self.state = TreeState.from_skeleton(sk0)
        self.light_ctx = LightContext(sk0, CFG.reward, seed=seed)
        self.r_tree = tree_horizontal_radius(sk0)
        self._refresh()
        self.k = compute_kpis(self.state.skeleton, self.props, CFG.reward, self.light_ctx)
        self.k_init = self.k
        self.total_r, self.steps = 0.0, 0
        self.trace: list[dict] = []

    def _refresh(self):
        self.props = compute_branch_props(self.state.skeleton, self.state.parent_map)
        self.crossing = detect_crossing(self.props, CFG.reward)

    def legal(self) -> list[int]:
        return self.state.legal_actions(CFG.data.min_nodes_abs, CFG.data.min_nodes_ratio)

    def must_stop(self, legal) -> bool:
        return (not legal or self.steps >= CFG.eval.max_steps
                or self.state.pruning_ratio() >= CFG.traj.max_pruning_ratio)

    def step(self, action: int) -> float:
        sk_before, props_before, k_before = self.state.skeleton, self.props, self.k
        crossing = self.crossing
        removed = sorted({action} | descendants(self.state.children, action))
        tags = branch_tags(action, props_before, crossing,
                           *branch_context(sk_before, props_before, CFG.reward), CFG.reward)
        self.state.remove_branch(action)
        self._refresh()
        self.k = compute_kpis(self.state.skeleton, self.props, CFG.reward, self.light_ctx)
        r, parts = composite_reward(action, sk_before, props_before, k_before,
                                    self.state.skeleton, self.props, self.k,
                                    crossing, CFG.reward)
        p = props_before.get(action)
        self.trace.append({
            "bid": int(action), "reward": float(r),
            **{k: float(v) for k, v in parts.items()},
            "tags": tags, "removed": [int(b) for b in removed],
            "points_removed": int(sk_before.shape[0] - self.state.skeleton.shape[0]),
            "pruning_ratio_after": float(self.state.pruning_ratio()),
            "kpi_after": {k: float(self.k[k]) for k in KPI_KEYS},
            "length": float(p.length) if p else None,
            "radius": float(p.avg_radius) if p else None,
        })
        self.total_r += r
        self.steps += 1
        return r

    def stop(self):
        self.total_r += stop_reward(self.state.pruning_ratio(), CFG.reward)

    def result(self, policy: str) -> dict:
        row = {"tree": tree_name(self.tree_dir), "policy": policy, "steps": self.steps,
               "total_reward": float(self.total_r),
               "final_pruning_ratio": self.state.pruning_ratio()}
        for key in KPI_KEYS:
            row[f"{key}_before"] = self.k_init[key]
            row[f"{key}_after"] = self.k[key]
            row[f"{key}_delta"] = self.k[key] - self.k_init[key]
        row["trace"] = self.trace
        return row


def rollout_baseline(tree_dir: str, policy: str, seed: int) -> dict:
    env = _Env(tree_dir, seed)
    rng = np.random.default_rng(seed)
    while True:
        legal = env.legal()
        if env.must_stop(legal):
            break
        if policy == "greedy":
            action = greedy_action(env.state, env.props, env.r_tree)
        elif policy == "expert":
            action = expert_action(env.state, env.props, env.crossing, env.light_ctx)
        else:
            action = int(rng.choice(legal))
        if action is None:
            break
        env.step(action)
    env.stop()
    return env.result(policy)


@torch.no_grad()
def rollout_learned(tree_dir: str, bundle: dict, target: dict | None,
                    seed: int, min_pruning_ratio: float = 0.0,
                    stop_logit_bias: float = 0.0) -> dict:
    """BC：逐步贪心取最高分；DT：自回归，RTG 随实得奖励递减（论文推理口径）。

    target：{"mode": "global", "value": 总回报} 或
            {"mode": "per_branch", "value": 每根初始分枝的回报系数}（目标 = 系数 × 初始分枝数）
    """
    kind, enc, model, meta, dev = (bundle[k] for k in
                                   ("kind", "encoder", "model", "meta", "device"))
    env = _Env(tree_dir, seed)
    if kind == "dt":
        K, scale = meta["context_len"], meta["rtg_scale"]
        target_return = target["value"] * (env.state.initial_branch_count
                                           if target["mode"] == "per_branch" else 1.0)
        rtg = target_return / scale
        hist_s, hist_a, hist_r, hist_t = [], [], [], []
    suppressed_stops = 0
    stop_margin = None
    stop_reason = "constraint"
    while True:
        legal = env.legal()
        if env.must_stop(legal):
            break
        arr = graph_arrays(env.state.skeleton, env.state.parent_map,
                           env.state.pruning_ratio())
        s_vec, nodes, _ = enc(Batch.from_data_list([arrays_to_data(arr)]).to(dev))
        N = nodes.size(1)
        legal_mask = torch.zeros(1, N, dtype=torch.bool, device=dev)
        legal_mask[0, torch.as_tensor(np.searchsorted(arr["bids"], legal), device=dev)] = True

        if kind != "dt":
            logits = model(s_vec, nodes, legal_mask)[0]
        else:
            hist_s.append(s_vec[0]); hist_r.append(rtg); hist_t.append(env.steps)
            S = torch.stack(hist_s[-K:]).unsqueeze(0)
            T = S.size(1)
            prev = hist_a[len(hist_a) - (T - 1):] if T > 1 else []
            acts = torch.stack(prev + [nodes.new_zeros(nodes.size(-1))])
            ctx = model(S, acts.unsqueeze(0),
                        torch.tensor(hist_r[-K:], device=dev).view(1, T, 1),
                        torch.tensor(hist_t[-K:], device=dev).view(1, T))
            logits = model.head(ctx[:, -1], nodes, legal_mask)[0]

        logits = logits.clone()
        if kind == "dt":
            logits[N] += stop_logit_bias
        branch_best = float(logits[:N][legal_mask[0]].max().item())
        stop_margin = float(logits[N].item() - branch_best)
        if env.state.pruning_ratio() < min_pruning_ratio:
            if int(torch.argmax(logits).item()) == N:
                suppressed_stops += 1
            if kind == "bcq":
                # 受控继续时，将 BCQ 的行为支持集条件化为“非 STOP”。
                # 否则 STOP 若是唯一通过阈值的动作，屏蔽它后会只剩 NEG_INF，
                # argmax 可能落到非法分枝。这里仍按原 tau 约束，只把归一化
                # 参照从全部动作改为合法分枝中的最大模仿 logit。
                q_logits = model.q(s_vec, nodes, legal_mask)[0]
                imit_logits = model.imit(s_vec, nodes, legal_mask)[0, :N]
                legal_imit = imit_logits[legal_mask[0]]
                allowed = legal_mask[0] & (
                    imit_logits - legal_imit.max() >= model.bcq_log_tau
                )
                logits[:N] = q_logits[:N].masked_fill(
                    ~allowed, torch.finfo(logits.dtype).min
                )
            logits[N] = torch.finfo(logits.dtype).min
        logits[:N] = logits[:N].masked_fill(
            ~legal_mask[0], torch.finfo(logits.dtype).min
        )
        idx = int(torch.argmax(logits).item())
        if idx == N:
            stop_reason = "model"
            break
        r = env.step(int(arr["bids"][idx]))
        if kind == "dt":
            hist_a.append(nodes[0, idx])
            # 记入结果的是完整奖励 r；从目标回报中扣除的必须与训练回报集相同
            reward_set = target.get("reward_set", "full")
            cond = r if reward_set == "full" else reward_for_set(env.trace[-1], reward_set)
            rtg -= cond / scale
    env.stop()
    row = env.result(kind)
    if kind == "dt":
        row["target_return"] = float(target_return)
    row["stop_reason"] = stop_reason
    row["stop_margin"] = stop_margin
    row["suppressed_stop_steps"] = suppressed_stops
    row["min_pruning_ratio"] = min_pruning_ratio
    row["stop_logit_bias"] = stop_logit_bias
    return row


def return_stats(reward_set: str = "full", policy: str = "expert") -> dict:
    """训练树（不含验证树）指定轨迹的总回报与每根初始分枝回报，按回报集缓存。"""
    from graph_cache import CACHE_ROOT, load_index, load_val_split
    if policy not in ("expert", "random"):
        raise SystemExit(f"未知目标轨迹类型 {policy}")
    filename = (f"expert_return_stats_{reward_set}.json" if policy == "expert"
                else f"return_stats_{reward_set}_{policy}.json")
    path = os.path.join(CACHE_ROOT, filename)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    index = load_index(reward_set)
    totals, per_branch = [], []
    for n in load_val_split()["train"]:
        chosen = [t["total_reward"] for t in index.get(n, []) if t["policy"] == policy]
        if not chosen:
            continue
        sk_ids = np.loadtxt(skeleton_path(os.path.join(CFG.data.data_root, n)), usecols=5)
        n0 = len(np.unique(sk_ids))
        totals += chosen
        per_branch += [r / n0 for r in chosen]
    if not totals:
        raise SystemExit(
            f"回报集 {reward_set} 没有训练树 {policy} 轨迹："
            f"请先运行 reward_parts.py apply --set {reward_set}"
        )
    stats = {"policy": policy, "totals": totals, "per_branch": per_branch}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f)
    return stats


def expert_return_stats(reward_set: str = "full") -> dict:
    return return_stats(reward_set, "expert")


def make_target(mode: str, quantile: float, value: float | None, reward_set: str = "full",
                target_policy: str = "expert") -> dict:
    """DT 初始目标回报设定（只用训练树中 target_policy 的统计）。"""
    if value is not None:
        return {"mode": mode, "value": value, "reward_set": reward_set,
                "target_policy": target_policy}
    stats = return_stats(reward_set, target_policy)
    key = "per_branch" if mode == "per_branch" else "totals"
    return {"mode": mode, "value": float(np.quantile(stats[key], quantile)),
            "reward_set": reward_set, "target_policy": target_policy}


# ==============================================================================
# 批量评估（多进程）
# ==============================================================================

_W = {}


def _init_worker(policy, checkpoint, target, device,
                 min_pruning_ratio=0.0, stop_logit_bias=0.0):
    torch.set_num_threads(1)
    _W.update(policy=policy, target=target,
              min_pruning_ratio=min_pruning_ratio,
              stop_logit_bias=stop_logit_bias,
              bundle=load_learned(checkpoint, torch.device(device))
              if policy in LEARNED else None)


def _run_one(tree_dir: str) -> dict:
    if _W["policy"] in LEARNED:
        return rollout_learned(tree_dir, _W["bundle"], _W["target"], CFG.eval.seed,
                               _W["min_pruning_ratio"], _W["stop_logit_bias"])
    return rollout_baseline(tree_dir, _W["policy"], CFG.eval.seed)


def run_policy(policy, checkpoint, trees, workers, target=None,
               min_pruning_ratio=0.0, stop_logit_bias=0.0) -> list[dict]:
    if policy == "dt":
        unit = "× 初始分枝数" if target["mode"] == "per_branch" else ""
        print(f"初始目标回报（{target['mode']}）：{target['value']:.3f} {unit}")
    device = "cpu" if (workers > 1 or not torch.cuda.is_available()) else "cuda"
    rows = []
    if workers <= 1:
        _init_worker(policy, checkpoint, target, device,
                     min_pruning_ratio, stop_logit_bias)
        it = map(_run_one, trees)
        pool = None
    else:
        pool = Pool(workers, initializer=_init_worker,
                    initargs=(policy, checkpoint, target, device,
                              min_pruning_ratio, stop_logit_bias))
        it = pool.imap(_run_one, trees)
    for i, row in enumerate(it, 1):
        rows.append(row)
        print(f"[{i}/{len(trees)}] {row['tree']} {policy}: reward={row['total_reward']:.2f} "
              f"steps={row['steps']} ratio={row['final_pruning_ratio']:.3f}", flush=True)
    if pool is not None:
        pool.close(); pool.join()
    return rows


def result_dir(stage: str, exp: str, split: str) -> str:
    return os.path.join(RESULTS_DIR, stage, exp, split)


def save_rows(out_dir: str, name: str, rows: list[dict], run_info: dict):
    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    traces = {r["tree"]: r.pop("trace") for r in rows}
    with open(os.path.join(out_dir, f"actions_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False)
    path = os.path.join(out_dir, f"eval_{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, f"eval_{name}.meta.json"), "w", encoding="utf-8") as f:
        json.dump({**run_info, "n_trees": len(rows)}, f, indent=2, ensure_ascii=False)
    print("已保存:", path)


def compare(d: str, ref: str):
    """ref 与其余各策略在共同树上做配对 Wilcoxon 检验。"""
    from scipy.stats import wilcoxon
    tables = {}
    for fn in os.listdir(d):
        if fn.startswith("eval_") and fn.endswith(".csv"):
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                tables[fn[5:-4]] = {r["tree"]: r for r in csv.DictReader(f)}
    if ref not in tables:
        raise SystemExit(f"缺 {ref} 的结果")
    report = {}
    for name, table in tables.items():
        if name == ref:
            continue
        common = sorted(set(tables[ref]) & set(table))
        if not common:
            continue
        a = np.array([float(tables[ref][t]["total_reward"]) for t in common])
        b = np.array([float(table[t]["total_reward"]) for t in common])
        stat = wilcoxon(a, b) if len(common) >= 8 and np.any(a != b) else None
        report[f"{ref}_vs_{name}"] = {
            "n": len(common), f"{ref}_mean": float(a.mean()), "other_mean": float(b.mean()),
            "diff_mean": float((a - b).mean()), "win_rate": float((a > b).mean()),
            "wilcoxon_p": float(stat.pvalue) if stat else None,
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(os.path.join(d, f"significance_{ref}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def split_dirs(split: str) -> list[str]:
    if split == "val":
        from graph_cache import load_val_split
        return [os.path.join(CFG.data.data_root, n) for n in load_val_split()["val"]]
    from trajectories import split_trees
    return split_trees(collect_tree_dirs(CFG.data.data_root))[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=[*LEARNED, "greedy", "expert", "random"],
                    default="greedy")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--limit", type=int, default=None, help="只评估该划分的前 N 棵树（固定顺序）")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--workers", type=int, default=1, help=">1 时多进程（模型改用 CPU 推理）")
    ap.add_argument("--target-mode", choices=["global", "per_branch"], default="per_branch",
                    help="DT 目标回报：global=全部树同一总回报；per_branch=系数×该树初始分枝数")
    ap.add_argument("--target-quantile", type=float, default=CFG.eval.target_return_quantile,
                    help="目标取训练树指定轨迹统计量的分位数")
    ap.add_argument("--target-policy", choices=["expert", "random"], default="expert",
                    help="目标回报使用的训练轨迹；去园艺知识消融必须用 random")
    ap.add_argument("--target-return", type=float, default=None,
                    help="直接指定目标值（global 为总回报，per_branch 为系数），覆盖分位数")
    ap.add_argument("--target-multiplier", type=float, default=1.0,
                    help="DT STOP 诊断：将统计或指定的初始目标回报乘以该系数")
    ap.add_argument("--min-pruning-ratio", type=float, default=0.0,
                    help="学习策略达到该剪枝比前屏蔽 STOP；用于受控选枝过程分析")
    ap.add_argument("--stop-logit-bias", type=float, default=0.0,
                    help="DT STOP 诊断：加到 STOP logit 的偏置；负值使模型更晚停止")
    ap.add_argument("--reward-set", type=str, default="full",
                    help="DT 目标回报统计所用的回报集（应与该 DT 训练时一致）")
    ap.add_argument("--tag", type=str, default=None, help="结果文件名后缀，如 dt_rtg800")
    ap.add_argument("--stage", choices=["dev", "paper"], default="dev",
                    help="dev=调参/过程数据；paper=写进论文的正式数据")
    ap.add_argument("--exp", type=str, default=None,
                    help="实验子目录名；paper 阶段必填（如 main、E2_teacher），dev 默认 adhoc")
    ap.add_argument("--overwrite", action="store_true", help="paper 阶段允许覆盖已有结果")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--ref", type=str, default="dt")
    args = ap.parse_args()

    if args.stage == "paper" and not args.exp:
        raise SystemExit("--stage paper 需要 --exp 指定实验编号")
    if args.split == "test" and args.stage != "paper":
        raise SystemExit("测试树只用于最终报告，需 --stage paper --exp <实验编号>")
    out_dir = result_dir(args.stage, args.exp or "adhoc", args.split)

    if args.compare:
        compare(out_dir, args.ref)
        return
    if args.policy in LEARNED and not args.checkpoint:
        raise SystemExit(f"--policy {args.policy} 需要 --checkpoint")
    if args.policy not in LEARNED and args.min_pruning_ratio != 0.0:
        raise SystemExit("--min-pruning-ratio 仅用于学习策略")
    if args.policy != "dt" and args.stop_logit_bias != 0.0:
        raise SystemExit("--stop-logit-bias 仅用于 DT 诊断")
    if args.policy != "dt" and args.target_multiplier != 1.0:
        raise SystemExit("--target-multiplier 仅用于 DT 诊断")
    if args.target_multiplier <= 0.0:
        raise SystemExit("--target-multiplier 必须大于 0")
    if not 0.0 <= args.min_pruning_ratio <= CFG.traj.max_pruning_ratio:
        raise SystemExit(f"--min-pruning-ratio 必须在 [0, {CFG.traj.max_pruning_ratio}]")
    name = args.tag or args.policy
    if (args.stage == "paper" and not args.overwrite
            and os.path.exists(os.path.join(out_dir, f"eval_{name}.csv"))):
        raise SystemExit(f"{out_dir} 已有 eval_{name}.csv；确需重跑请加 --overwrite")
    trees = split_dirs(args.split)
    if args.only:
        trees = [d for d in trees if tree_name(d) == args.only]
    if args.limit:
        trees = trees[:args.limit]
    if args.policy == "dt":
        ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) \
            else os.path.join(CFG.data.work_root, args.checkpoint)
        meta_path = os.path.join(os.path.dirname(ckpt_path), "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                trained = json.load(f)
            trained_set = trained.get("reward_set", "full")
            if trained_set != args.reward_set:
                raise SystemExit(
                    f"checkpoint 使用回报集 {trained_set}，评估却指定 {args.reward_set}"
                )
            trained_policies = trained.get("policies")
            if trained_policies and "expert" not in trained_policies and args.target_policy != "random":
                raise SystemExit("该模型未使用专家轨迹，目标回报必须加 --target-policy random")
    target = (make_target(args.target_mode, args.target_quantile, args.target_return,
                          args.reward_set, args.target_policy)
              if args.policy == "dt" else None)
    if target is not None:
        target["value"] *= args.target_multiplier
        target["multiplier"] = args.target_multiplier
    rows = run_policy(args.policy, args.checkpoint, trees, args.workers, target,
                      args.min_pruning_ratio, args.stop_logit_bias)
    run_info = {"command": " ".join(sys.argv), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "policy": args.policy, "checkpoint": args.checkpoint, "target": target,
                "split": args.split, "stage": args.stage,
                "min_pruning_ratio": args.min_pruning_ratio,
                "stop_logit_bias": args.stop_logit_bias}
    save_rows(out_dir, name, rows, run_info)


if __name__ == "__main__":
    main()
