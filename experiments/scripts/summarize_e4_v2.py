"""汇总 v2 E4 五个核心消融，并进行配对检验。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "dev" / "e4_redesign_v2" / "val"
SCALES = ROOT / "data" / "graph_cache_v2_curated" / "kpi_scales.json"
VARIANTS = ("full", "no_context", "no_gardening", "no_rq", "no_kpi", "no_rc")
ABLATIONS = VARIANTS[1:]
KPI = ("hierarchy", "crowding", "balance", "light", "potential", "stability")
WEIGHTS = {
    "hierarchy": 5.0, "crowding": 10.0, "balance": 8.0,
    "light": 8.0, "potential": 2.0, "stability": 5.0,
}
DEFECT = {"cross_inferior", "inward", "vigorous", "drooping", "weak"}


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return {r["tree"]: r for r in csv.DictReader(f)}


def kpi_score(values, scales):
    return sum(WEIGHTS[k] * values[k] / scales[k] for k in KPI)


def bootstrap_ci(diff, seed=42, n_boot=10000):
    rng = np.random.default_rng(seed)
    draws = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def paired_test(full, ablated):
    diff = np.asarray(full) - np.asarray(ablated)
    nonzero = diff != 0
    if nonzero.any():
        p = float(wilcoxon(diff[nonzero], alternative="two-sided").pvalue)
        ranks = rankdata(np.abs(diff[nonzero]))
        rbc = float(
            (ranks[diff[nonzero] > 0].sum() - ranks[diff[nonzero] < 0].sum())
            / ranks.sum())
    else:
        p, rbc = 1.0, 0.0
    return {
        "mean_difference_full_minus_ablation": float(diff.mean()),
        "ci95": bootstrap_ci(diff),
        "p": p,
        "rank_biserial": rbc,
        "full_win_rate": float(np.mean(diff > 0)),
    }


def holm(tests):
    ordered = sorted(tests, key=lambda x: x[1]["p"])
    running = 0.0
    m = len(ordered)
    for i, (_, test) in enumerate(ordered):
        running = max(running, (m - i) * test["p"])
        test["p_holm"] = float(min(running, 1.0))


def main():
    with SCALES.open(encoding="utf-8") as f:
        scales = json.load(f)["scales"]
    data = {}
    for variant in VARIANTS:
        endpoint = read_csv(RESULT / f"eval_{variant}.csv")
        curve = read_csv(RESULT / f"eval_{variant}_curve.csv")
        with (RESULT / f"actions_{variant}.json").open(encoding="utf-8") as f:
            actions = json.load(f)
        with (RESULT / f"actions_{variant}_curve.json").open(encoding="utf-8") as f:
            curve_actions = json.load(f)
        trees = sorted(endpoint)
        metrics = {k: [] for k in (
            "ratio", "full_reward", "kpi_terminal", "kpi_cap", "kpi_auc",
            "rq_per_cut", "rc_per_cut", "defect_rate", "protect_rate",
            "main_branch_rate", "model_stop")}
        for tree in trees:
            row, cap = endpoint[tree], curve[tree]
            metrics["ratio"].append(float(row["final_pruning_ratio"]))
            metrics["full_reward"].append(float(row["total_reward_full"]))
            metrics["model_stop"].append(float(row["stop_reason"] == "model"))
            metrics["kpi_terminal"].append(kpi_score(
                {k: float(row[f"{k}_delta"]) for k in KPI}, scales))
            metrics["kpi_cap"].append(kpi_score(
                {k: float(cap[f"{k}_delta"]) for k in KPI}, scales))
            trace = curve_actions[tree]
            x = [0.0] + [float(s["pruning_ratio_after"]) for s in trace]
            y = [0.0] + [
                kpi_score({
                    k: float(s["kpi_after"][k]) - float(cap[f"{k}_before"])
                    for k in KPI
                }, scales) for s in trace]
            metrics["kpi_auc"].append(float(np.trapz(y, x) / 0.32))
            endpoint_trace = actions[tree]
            n = max(len(endpoint_trace), 1)
            metrics["rq_per_cut"].append(
                sum(float(s["rq"]) for s in endpoint_trace) / n)
            metrics["rc_per_cut"].append(
                sum(float(s["rc"]) for s in endpoint_trace) / n)
            metrics["defect_rate"].append(
                sum(bool(DEFECT & set(s["tags"])) for s in endpoint_trace) / n)
            metrics["protect_rate"].append(
                sum(any(t.startswith("protect_") for t in s["tags"])
                    for s in endpoint_trace) / n)
            metrics["main_branch_rate"].append(
                sum("main_branch" in s["tags"] for s in endpoint_trace) / n)
        data[variant] = {k: np.asarray(v, dtype=float) for k, v in metrics.items()}

    means = {
        variant: {metric: float(values.mean()) for metric, values in rows.items()}
        for variant, rows in data.items()
    }
    tests = {}
    for metric in data["full"]:
        tests[metric] = {
            variant: paired_test(data["full"][metric], data[variant][metric])
            for variant in ABLATIONS
        }
        holm(list(tests[metric].items()))
    output = {"n_trees": 31, "means": means, "paired_tests": tests}
    with (RESULT / "e4_v2_summary.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
