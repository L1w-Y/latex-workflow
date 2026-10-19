"""在冻结 DT 自身 rollout 状态上收集净收益 oracle 的 STOP 监督。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dt_pruning"))

from config import CFG  # noqa: E402
from ablation_v2 import mask_graph_features  # noqa: E402
from encoder import arrays_to_data, graph_arrays  # noqa: E402
from evaluate_v2 import EnvV2, load_bundle, target_return  # noqa: E402
from reward import reward_for_set  # noqa: E402
from trajectories_v2 import evaluate_candidates, stable_seed  # noqa: E402
from tree_data import collect_tree_dirs, tree_name  # noqa: E402
from v2_manifest import load_manifest, manifest_hash  # noqa: E402


@torch.no_grad()
def collect_tree(tree_dir, bundle, target, candidate_cap):
    encoder, model, meta, device = bundle
    reward_set = meta.get("reward_set", "full")
    env = EnvV2(tree_dir, CFG.eval.seed, reward_set)
    rng = np.random.default_rng(stable_seed(tree_name(tree_dir), "stop_onpolicy", 0))
    context_len, scale = meta["context_len"], meta["rtg_scale"]
    rtg = target / scale
    hist_s, hist_a, hist_r, hist_t = [], [], [], []
    features, labels, ratios, oracle_net = [], [], [], []
    low = 0
    reason = "constraint"
    while env.steps < CFG.eval.max_steps:
        legal = env.legal()
        if not legal:
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
        legal_mask[0, torch.as_tensor(
            np.searchsorted(arr["bids"], legal), device=device)] = True
        hist_s.append(state[0]); hist_r.append(rtg); hist_t.append(env.steps)
        states = torch.stack(hist_s[-context_len:]).unsqueeze(0)
        length = states.size(1)
        previous = hist_a[len(hist_a) - (length - 1):] if length > 1 else []
        actions = torch.stack(
            previous + [nodes.new_zeros(nodes.size(-1))]).unsqueeze(0)
        ctx = model(
            states, actions,
            torch.tensor(hist_r[-context_len:], device=device).view(1, length, 1),
            torch.tensor(hist_t[-context_len:], device=device).view(1, length))
        branch, _, _, _ = model.head(ctx[:, -1], nodes, legal_mask)
        stop_input, _ = model.head.stop_features(ctx[:, -1], nodes, legal_mask)
        idx = int(branch[0].argmax().item())
        model_bid = int(arr["bids"][idx])

        records = evaluate_candidates(
            env.state, env.props, env.k, env.crossing, env.ctx, rng,
            candidate_cap=candidate_cap, forced_bids=[model_bid])
        for record in records:
            record["net"] = reward_for_set(
                record["parts"], reward_set, CFG.reward)
        best_net = max((x["net"] for x in records), default=float("-inf"))
        low = low + 1 if best_net <= CFG.traj.v2_net_gain_threshold else 0
        should_stop = (
            env.state.pruning_ratio() >= CFG.traj.v2_min_stop_ratio
            and low >= CFG.traj.v2_stop_patience)
        features.append(stop_input[0].cpu().numpy().astype(np.float32))
        labels.append(float(should_stop))
        ratios.append(float(env.state.pruning_ratio()))
        oracle_net.append(float(best_net))
        if should_stop:
            reason = "oracle"
            break
        hist_a.append(nodes[0, idx])
        reward = env.step(model_bid)
        rtg -= reward / scale
    return {
        "x": np.asarray(features, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.float32),
        "ratio": np.asarray(ratios, dtype=np.float32),
        "oracle_net": np.asarray(oracle_net, dtype=np.float32),
        "reason": reason,
        "final_ratio": float(env.state.pruning_ratio()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate-cap", type=int, default=8)
    ap.add_argument("--out", default="data/stop_onpolicy_v2")
    ap.add_argument("--max-trees", type=int, default=None,
                    help="冒烟：每个 split 只处理前N棵")
    args = ap.parse_args()
    checkpoint = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(
        CFG.data.work_root, args.checkpoint)
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    loaded = load_bundle(checkpoint, device)
    bundle = (loaded[0], loaded[1], loaded[2], device)
    target = target_return(loaded[2], 1.0)
    manifest = load_manifest()
    dirs = {tree_name(d): d for d in collect_tree_dirs(CFG.data.data_root)}
    root = Path(args.out)
    if not root.is_absolute():
        root = Path(CFG.data.work_root) / root
    root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val"):
        records, tree_rows = [], []
        names = (manifest[split][:args.max_trees]
                 if args.max_trees else manifest[split])
        for i, name in enumerate(names, 1):
            rec = collect_tree(
                dirs[name], bundle, target, args.candidate_cap)
            records.append(rec)
            tree_rows.append({
                "tree": name, "states": len(rec["y"]),
                "positives": int(rec["y"].sum()),
                "reason": rec["reason"], "final_ratio": rec["final_ratio"],
            })
            print(
                f"[{split} {i}/{len(names)}] {name}: "
                f"states={len(rec['y'])} positive={int(rec['y'].sum())} "
                f"ratio={rec['final_ratio']:.3f} {rec['reason']}", flush=True)
        x = np.concatenate([r["x"] for r in records])
        y = np.concatenate([r["y"] for r in records])
        ratio = np.concatenate([r["ratio"] for r in records])
        oracle_net = np.concatenate([r["oracle_net"] for r in records])
        np.savez_compressed(
            root / f"{split}.npz", x=x, y=y, ratio=ratio, oracle_net=oracle_net)
        summary[split] = {
            "trees": tree_rows, "states": int(len(y)),
            "positives": int(y.sum()), "feature_dim": int(x.shape[1]),
        }
    meta = {
        "checkpoint": checkpoint, "manifest_hash": manifest_hash(manifest),
        "candidate_cap": args.candidate_cap, "target_return": target,
        "net_gain_threshold": CFG.traj.v2_net_gain_threshold,
        "stop_patience": CFG.traj.v2_stop_patience, "splits": summary,
    }
    with (root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("写入", root)


if __name__ == "__main__":
    main()
