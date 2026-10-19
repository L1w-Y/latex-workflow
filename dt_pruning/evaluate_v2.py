"""schema=2 DT 验证集评估；独立停止头直接决定自主 STOP。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from ablation_v2 import mask_graph_features
from config import CFG
from encoder import arrays_to_data, graph_arrays
from kpi_scales_v2 import activate_v2_kpi_scales
from reward import (KPI_KEYS, TreeContext, branch_context, branch_tags,
                    composite_reward_v2, compute_branch_props, compute_kpis,
                    detect_crossing, reward_for_set)
from train_v2 import build_models_from_meta
from tree_data import (TreeState, collect_tree_dirs, descendants, load_skeleton,
                       skeleton_path, tree_name)
from v2_manifest import load_manifest, manifest_hash


class EnvV2:
    def __init__(self, tree_dir: str, seed: int, reward_set: str = "full"):
        sk0 = load_skeleton(skeleton_path(tree_dir))
        self.tree_dir = tree_dir
        self.state = TreeState.from_skeleton(sk0)
        self.ctx = TreeContext(sk0, CFG.reward, seed=seed)
        self.props = compute_branch_props(sk0, self.state.parent_map)
        self.k = compute_kpis(sk0, self.props, CFG.reward, self.ctx)
        self.k_init = dict(self.k)
        self.crossing = detect_crossing(self.props, CFG.reward)
        self.steps = 0
        self.total_reward = 0.0
        self.total_reward_full = 0.0
        self.reward_set = reward_set
        self.trace = []

    def legal(self):
        legal = self.state.legal_actions(
            CFG.data.min_nodes_abs, CFG.data.min_nodes_ratio)
        # 与轨迹生成一致：执行后会越过安全上限的动作不能进入模型动作集。
        safe = []
        for bid in legal:
            removed = 1 + len(descendants(self.state.children, bid))
            ratio_after = (
                self.state.initial_branch_count
                - (len(self.state.parent_map) - removed)
            ) / max(self.state.initial_branch_count, 1)
            if ratio_after <= CFG.traj.v2_max_pruning_ratio + 1e-9:
                safe.append(bid)
        return safe

    def step(self, bid: int):
        before, props_before, k_before, crossing = (
            self.state.skeleton, self.props, self.k, self.crossing)
        removed = sorted({bid} | descendants(self.state.children, bid))
        tags = branch_tags(
            bid, props_before, crossing,
            *branch_context(before, props_before, CFG.reward), CFG.reward)
        self.state.remove_branch(bid)
        self.props = compute_branch_props(self.state.skeleton, self.state.parent_map)
        self.k = compute_kpis(self.state.skeleton, self.props, CFG.reward, self.ctx)
        self.crossing = detect_crossing(self.props, CFG.reward)
        reward_full, parts = composite_reward_v2(
            bid, before, props_before, k_before,
            self.state.skeleton, self.props, self.k, crossing, CFG.reward)
        reward = reward_for_set(parts, self.reward_set, CFG.reward)
        self.total_reward += reward
        self.total_reward_full += reward_full
        self.steps += 1
        self.trace.append({
            "bid": int(bid), "reward": float(reward),
            **{k: float(v) for k, v in parts.items()},
            "tags": tags, "removed": [int(x) for x in removed],
            "pruning_ratio_after": float(self.state.pruning_ratio()),
            "kpi_after": {k: float(self.k[k]) for k in KPI_KEYS},
        })
        return reward

    def result(self, stop_reason: str, stop_logit: float | None):
        row = {
            "tree": tree_name(self.tree_dir), "policy": "dt_v2",
            "steps": self.steps, "total_reward": float(self.total_reward),
            "total_reward_full": float(self.total_reward_full),
            "final_pruning_ratio": float(self.state.pruning_ratio()),
            "stop_reason": stop_reason, "stop_logit": stop_logit,
        }
        for key in KPI_KEYS:
            row[f"{key}_before"] = self.k_init[key]
            row[f"{key}_after"] = self.k[key]
            row[f"{key}_delta"] = self.k[key] - self.k_init[key]
        row["trace"] = self.trace
        return row


def load_bundle(path: str, device):
    runtime_scales = activate_v2_kpi_scales()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "dt_v2":
        raise ValueError("evaluate_v2 只接受 kind=dt_v2 checkpoint")
    meta = checkpoint["meta"]
    stored_scales = meta.get("kpi_scales")
    if stored_scales and any(
            not np.isclose(float(stored_scales[k]), runtime_scales[k], rtol=1e-7)
            for k in runtime_scales):
        raise ValueError("checkpoint KPI尺度与当前唯一尺度文件不一致")
    manifest = load_manifest()
    if meta.get("schema_version") != 2 or meta.get("manifest_hash") != manifest_hash(manifest):
        raise ValueError("checkpoint 的 schema/manifest 与当前 v2 数据不一致")
    encoder, model = build_models_from_meta(device, meta)
    encoder.load_state_dict(checkpoint["encoder"])
    model.load_state_dict(checkpoint["model"])
    encoder.eval(); model.eval()
    return encoder, model, meta


def target_return(meta: dict, quantile: float) -> float:
    from graph_cache_v2 import load_cache, load_split
    cache_root = meta.get("cache_root")
    if cache_root and not os.path.isabs(cache_root):
        cache_root = os.path.join(CFG.data.work_root, cache_root)
    trajectories = load_cache(
        load_split()["train"], tuple(meta["policies"]),
        meta.get("reward_set", "full"), cache_root)
    totals = [t["total_reward"] for t in trajectories]
    if not totals:
        raise ValueError("训练缓存中没有目标回报统计")
    return float(np.quantile(totals, quantile))


@torch.no_grad()
def rollout(tree_dir: str, encoder, model, meta, device,
            target: float, stop_threshold: float,
            diagnose_stop_continuation: bool = False):
    env = EnvV2(tree_dir, CFG.eval.seed, meta.get("reward_set", "full"))
    context_len, scale = meta["context_len"], meta["rtg_scale"]
    rtg = target / scale
    hist_s, hist_a, hist_r, hist_t = [], [], [], []
    stop_reason, last_stop = "constraint", None
    stop_snapshot = None
    while True:
        legal = env.legal()
        if not legal or env.steps >= CFG.eval.max_steps:
            stop_reason = "constraint"
            break
        if env.state.pruning_ratio() >= CFG.traj.v2_max_pruning_ratio:
            stop_reason = "safety_cap"
            break
        arr = graph_arrays(
            env.state.skeleton, env.state.parent_map, env.state.pruning_ratio(),
            schema_version=2, tree_context=env.ctx, kpis=env.k)
        graph = Batch.from_data_list([arrays_to_data(arr)]).to(device)
        mask_graph_features(
            graph,
            use_rule_features=meta.get("use_rule_features", True),
            use_kpi_features=meta.get("use_kpi_features", True))
        state, nodes, _ = encoder(graph)
        n_nodes = nodes.size(1)
        legal_mask = torch.zeros(1, n_nodes, dtype=torch.bool, device=device)
        indices = np.searchsorted(arr["bids"], legal)
        legal_mask[0, torch.as_tensor(indices, device=device)] = True
        hist_s.append(state[0]); hist_r.append(rtg); hist_t.append(env.steps)
        states = torch.stack(hist_s[-context_len:]).unsqueeze(0)
        length = states.size(1)
        previous = hist_a[len(hist_a) - (length - 1):] if length > 1 else []
        acts = torch.stack(previous + [nodes.new_zeros(nodes.size(-1))]).unsqueeze(0)
        ctx = model(
            states, acts,
            torch.tensor(hist_r[-context_len:], device=device).view(1, length, 1),
            torch.tensor(hist_t[-context_len:], device=device).view(1, length))
        branch_logits, stop_logits, _, _ = model.head(ctx[:, -1], nodes, legal_mask)
        last_stop = float(stop_logits.item())
        if (stop_snapshot is None
                and torch.sigmoid(stop_logits).item() >= stop_threshold):
            stop_reason = "model"
            if not diagnose_stop_continuation:
                break
            stop_snapshot = {
                "steps": env.steps,
                "reward": float(env.total_reward),
                "ratio": float(env.state.pruning_ratio()),
                "kpis": dict(env.k),
                "stop_logit": last_stop,
            }
        idx = int(branch_logits[0].argmax().item())
        hist_a.append(nodes[0, idx])
        reward = env.step(int(arr["bids"][idx]))
        rtg -= reward / scale
    row = env.result(stop_reason, last_stop)
    row["target_return"] = target
    row["stop_threshold"] = stop_threshold
    if diagnose_stop_continuation:
        row["model_stopped"] = stop_snapshot is not None
        row["stop_steps"] = stop_snapshot["steps"] if stop_snapshot else ""
        row["stop_ratio"] = stop_snapshot["ratio"] if stop_snapshot else ""
        row["stop_reward"] = stop_snapshot["reward"] if stop_snapshot else ""
        row["continued_steps"] = (
            env.steps - stop_snapshot["steps"] if stop_snapshot else 0)
        row["continued_reward"] = (
            env.total_reward - stop_snapshot["reward"] if stop_snapshot else 0.0)
        for key in KPI_KEYS:
            stop_value = stop_snapshot["kpis"][key] if stop_snapshot else ""
            row[f"{key}_at_stop"] = stop_value
            row[f"{key}_continued_delta"] = (
                env.k[key] - stop_value if stop_snapshot else 0.0)
    return row


def save(rows: list[dict], out_dir: str, tag: str, info: dict):
    os.makedirs(out_dir, exist_ok=True)
    traces = {row["tree"]: row.pop("trace") for row in rows}
    with open(os.path.join(out_dir, f"actions_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False)
    with open(os.path.join(out_dir, f"eval_{tag}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(out_dir, f"eval_{tag}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=("val", "model_val"), default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--target-quantile", type=float, default=1.0)
    ap.add_argument("--target-return", type=float, default=None)
    ap.add_argument("--stop-threshold", type=float, default=0.5)
    ap.add_argument("--exp", default="e3_redesign_v2")
    ap.add_argument("--tag", default="dt_v2")
    ap.add_argument(
        "--diagnose-stop-continuation", action="store_true",
        help="首次模型STOP后继续执行同一Branch策略至安全上限，记录剩余收益/KPI")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not 0 < args.stop_threshold < 1:
        raise SystemExit("--stop-threshold 必须位于 (0,1)")
    checkpoint = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(
        CFG.data.work_root, args.checkpoint)
    out_dir = os.path.join(
        CFG.data.work_root, "results", "dev", args.exp, args.split)
    out_file = os.path.join(out_dir, f"eval_{args.tag}.csv")
    if os.path.exists(out_file) and not args.overwrite:
        raise SystemExit(f"结果已存在：{out_file}；确需覆盖请加 --overwrite")
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    encoder, model, meta = load_bundle(checkpoint, device)
    target = args.target_return if args.target_return is not None else target_return(
        meta, args.target_quantile)
    manifest = load_manifest()
    names = (
        manifest["val"] if args.split == "model_val"
        else manifest.get("eval_val", manifest["val"]))
    if args.only:
        names = [n for n in names if n == args.only]
    if args.limit:
        names = names[:args.limit]
    dirs = {tree_name(d): d for d in collect_tree_dirs(CFG.data.data_root)}
    rows, started = [], time.time()
    for i, name in enumerate(names, 1):
        row = rollout(
            dirs[name], encoder, model, meta, device, target, args.stop_threshold,
            args.diagnose_stop_continuation)
        rows.append(row)
        extra = ""
        if args.diagnose_stop_continuation and row["model_stopped"]:
            extra = (
                f" stop_ratio={row['stop_ratio']:.3f}"
                f" residual={row['continued_reward']:.2f}")
        print(f"[{i}/{len(names)}] {name}: ratio={row['final_pruning_ratio']:.3f} "
              f"reward={row['total_reward']:.2f} stop={row['stop_reason']}{extra}",
              flush=True)
    save(rows, out_dir, args.tag, {
        "command": " ".join(sys.argv), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": checkpoint, "target_return": target,
        "stop_threshold": args.stop_threshold, "n_trees": len(rows),
        "diagnose_stop_continuation": args.diagnose_stop_continuation,
        "minutes": (time.time() - started) / 60,
    })
    print(f"结果：{out_file}")


if __name__ == "__main__":
    main()
