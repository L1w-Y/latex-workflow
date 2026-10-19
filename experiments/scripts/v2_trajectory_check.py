"""快速检查已生成的 v2 轨迹，不重新计算 KPI。"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dt_pruning"))

from v2_manifest import V2_TRAJ_ROOT, load_manifest  # noqa: E402


def main() -> None:
    manifest = load_manifest()
    files = sorted(V2_TRAJ_ROOT.glob("*.pkl"))
    train_set, val_set = set(manifest["train"]), set(manifest["val"])
    report = {
        "files": len(files),
        "from_train_pool": sum(p.stem in train_set for p in files),
        "from_eval_val": sum(p.stem in val_set for p in files),
        "unknown": [p.stem for p in files if p.stem not in train_set | val_set],
    }
    policies, stops = Counter(), Counter()
    steps, ratios, rewards, bad = [], [], [], []
    best_net, rq, ri, rc, cost = [], [], [], [], []
    best_net_by_ratio = {key: [] for key in ("<.15", ".15-.20", ".20-.25", ".25-.28", ".28-.30", ">=.30")}
    cost_grid = {m: [] for m in (1000, 1200, 1400, 1600, 1800, 2000)}
    threshold_grid = {x: [] for x in (60, 80, 100, 120, 140)}
    names = []
    for path in files:
        try:
            with path.open("rb") as f:
                trajs = pickle.load(f)
            names.append(path.stem)
            if len(trajs) != 5:
                bad.append(f"{path.stem}: trajectories={len(trajs)}")
            for tr in trajs:
                policies[tr["policy"]] += 1
                stops[tr["metadata"]["stop_reason"]] += 1
                steps.append(len(tr["actions"]))
                ratios.append(float(tr["metadata"]["final_pruning_ratio"]))
                rewards.append(float(tr["metadata"]["total_reward"]))
                best_net.extend(x for x in tr["metadata"]["best_net_history"] if np.isfinite(x))
                _collect_best_net_by_ratio(tr, best_net_by_ratio)
                for part, action in zip(tr["reward_parts"], tr["actions"]):
                    if action < 0:
                        continue
                    rq.append(float(part["rq"]))
                    ri.append(float(part["ri"]))
                    rc.append(float(part["rc"]))
                    cost.append(float(part["cost"]))
                _append_hypothetical_stops(tr, cost_grid)
                _append_threshold_stops(tr, threshold_grid)
                if len(tr["actions"]) != len(tr["rewards"]) or tr["actions"][-1] != -1:
                    bad.append(f"{path.stem}/{tr['policy']}: alignment")
        except Exception as exc:
            bad.append(f"{path.stem}: {exc!r}")
    report.update({
        "names": names,
        "policies": dict(policies),
        "stop_reasons": dict(stops),
        "steps": _stats(steps),
        "final_pruning_ratio": _stats(ratios),
        "total_reward": _stats(rewards),
        "best_candidate_net": _quantiles(best_net),
        "best_candidate_net_by_ratio": {
            key: _quantiles(values) for key, values in best_net_by_ratio.items()},
        "executed_parts": {
            "alpha_rq": _quantiles(np.asarray(rq) * 5.0),
            "beta_ri": _quantiles(np.asarray(ri) * 3.0),
            "gamma_rc": _quantiles(rc),
            "prune_cost": _quantiles(cost),
        },
        "cost_multiplier_stop_ratio": {
            str(m): _stats(values) for m, values in cost_grid.items()},
        "fixed_action_cost_stop_ratio": {
            str(x): _stats(values) for x, values in threshold_grid.items()},
        "problems": bad,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _stats(values) -> dict:
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {}
    return {
        "min": float(x.min()), "median": float(np.median(x)),
        "mean": float(x.mean()), "max": float(x.max()),
    }


def _quantiles(values) -> dict:
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {}
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)}


def _append_hypothetical_stops(tr: dict, grid: dict) -> None:
    n0 = tr["metadata"]["initial_branch_count"]
    ratios = [
        (n0 - len(np.unique(sk[:, 5].astype(np.int64)))) / max(n0, 1)
        for sk in tr["skeletons"]
    ]
    parts = tr["reward_parts"]
    old_rewards = tr["rewards"]
    for multiplier, values in grid.items():
        low = 0
        stop_ratio = ratios[-1]
        for t, (part, action) in enumerate(zip(parts, tr["actions"])):
            if action < 0:
                break
            adjusted = old_rewards[t] - (multiplier - 1.0) * float(part["cost"])
            low = low + 1 if adjusted <= 0 else 0
            if ratios[t] >= 0.12 and low >= 2:
                stop_ratio = ratios[t]
                break
        values.append(float(stop_ratio))


def _collect_best_net_by_ratio(tr: dict, output: dict) -> None:
    n0 = tr["metadata"]["initial_branch_count"]
    for sk, net in zip(tr["skeletons"], tr["metadata"]["best_net_history"]):
        if not np.isfinite(net):
            continue
        ratio = (n0 - len(np.unique(sk[:, 5].astype(np.int64)))) / max(n0, 1)
        key = ("<.15" if ratio < .15 else ".15-.20" if ratio < .20
               else ".20-.25" if ratio < .25 else ".25-.28" if ratio < .28
               else ".28-.30" if ratio < .30 else ">=.30")
        output[key].append(float(net))


def _append_threshold_stops(tr: dict, grid: dict) -> None:
    n0 = tr["metadata"]["initial_branch_count"]
    ratios = [
        (n0 - len(np.unique(sk[:, 5].astype(np.int64)))) / max(n0, 1)
        for sk in tr["skeletons"]
    ]
    for threshold, values in grid.items():
        low, stop_ratio = 0, ratios[-1]
        for t, net in enumerate(tr["metadata"]["best_net_history"]):
            low = low + 1 if net <= threshold else 0
            if ratios[min(t, len(ratios) - 1)] >= .12 and low >= 2:
                stop_ratio = ratios[min(t, len(ratios) - 1)]
                break
        values.append(float(stop_ratio))


if __name__ == "__main__":
    main()
