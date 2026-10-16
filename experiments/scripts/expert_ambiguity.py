"""专家决策歧义度 + BC 选择质量分析（验证树）。

对验证树专家轨迹抽样若干步，统计：
- 专家打分的并列程度：分数 ≥ 最高分 95% / 90% 的候选枝数
- BC top-1 选中枝在专家打分中的名次、相对最高分的比例、是否为正分（专家认为该剪）

用法（在 dt_pruning 目录下）：
    python ../experiments/scripts/expert_ambiguity.py --checkpoint checkpoints_bc/bc_best.pt --trees 8 --every 4
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))

from config import CFG  # noqa: E402
from encoder import arrays_to_data, graph_arrays  # noqa: E402
from evaluate import load_learned  # noqa: E402
from graph_cache import TRAJ_ROOT, load_val_split  # noqa: E402
from reward import compute_branch_props, detect_crossing  # noqa: E402
from trajectories import expert_scores  # noqa: E402
from tree_data import BRANCH, TreeState  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_bc/bc_best.pt")
    ap.add_argument("--trees", type=int, default=8)
    ap.add_argument("--every", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_learned(args.checkpoint, device)
    enc, model = bundle["encoder"], bundle["model"]

    rows = []
    for name in load_val_split()["val"][:args.trees]:
        with open(os.path.join(TRAJ_ROOT, f"{name}.pkl"), "rb") as f:
            tr = next(t for t in pickle.load(f) if t["policy"] == "expert")
        sk0 = tr["skeletons"][0]
        n0 = len(np.unique(sk0[:, BRANCH]))
        for t in range(0, len(tr["actions"]), args.every):
            a = tr["actions"][t]
            if a < 0:
                continue
            sk, pm = tr["skeletons"][t], tr["parent_maps"][t]
            state = TreeState(skeleton=sk, initial_nodes=sk0.shape[0], initial_branch_count=n0)
            props = compute_branch_props(sk, state.parent_map)
            scores = expert_scores(state, props, detect_crossing(props, CFG.reward))
            bids_sorted = sorted(scores, key=scores.get, reverse=True)
            s_max = scores[bids_sorted[0]]

            m = len(np.unique(sk[:, BRANCH]))
            arr = graph_arrays(sk, pm, (n0 - m) / n0)
            with torch.no_grad():
                s_vec, nodes, _ = enc(Batch.from_data_list([arrays_to_data(arr)]).to(device))
                legal = torch.zeros(1, nodes.size(1), dtype=torch.bool, device=device)
                legal[0, torch.as_tensor(np.searchsorted(arr["bids"], list(scores)), device=device)] = True
                logits = model(s_vec, nodes, legal)[0, :-1]
            pick = int(arr["bids"][int(torch.argmax(logits))])
            s_pick = scores.get(pick, float("nan"))
            rows.append({
                "tree": name, "t": t, "n_legal": len(scores), "s_max": s_max,
                "n_ge95": sum(s >= 0.95 * s_max for s in scores.values()),
                "n_ge90": sum(s >= 0.90 * s_max for s in scores.values()),
                "n_pos": sum(s > 0 for s in scores.values()),
                "expert_is_top": scores.get(a) == s_max,
                "bc_rank": bids_sorted.index(pick) + 1 if pick in scores else None,
                "bc_ratio": s_pick / s_max if s_max > 0 else None,
                "bc_positive": s_pick > 0,
            })
        print(f"{name}: 已抽样 {len(rows)} 步", flush=True)

    def q(key):
        v = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
        return {"median": float(np.median(v)), "p25": float(np.quantile(v, .25)),
                "p75": float(np.quantile(v, .75)), "mean": float(v.mean())}

    ranks = np.array([r["bc_rank"] for r in rows if r["bc_rank"] is not None])
    summary = {
        "n_states": len(rows),
        "legal_candidates": q("n_legal"),
        "positive_score_candidates": q("n_pos"),
        "candidates_ge95pct_of_max": q("n_ge95"),
        "candidates_ge90pct_of_max": q("n_ge90"),
        "expert_label_is_argmax_rate": float(np.mean([r["expert_is_top"] for r in rows])),
        "bc_pick_rank": q("bc_rank"),
        "bc_pick_in_top10_rate": float(np.mean(ranks <= 10)),
        "bc_pick_score_ratio_to_max": q("bc_ratio"),
        "bc_pick_positive_rate": float(np.mean([r["bc_positive"] for r in rows])),
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    out = os.path.join(CFG.data.work_root, "results", "dev", "pilot", "val", "expert_ambiguity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
