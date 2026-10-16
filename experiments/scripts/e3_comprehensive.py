"""E3 多维综合基线分析。

终点结果：
- DT 读取 stop_diag/val/eval_dt_bias_m06.*
- 其余策略读取 e3_v2/val/eval_<policy>.*

过程结果：
- 读取 e3_comprehensive/val/eval_curve_<policy>.*
- 学习策略在剪枝比 0.30 前屏蔽 STOP，用于过程曲线与同剪枝量比较

输出全部写入 results/<stage>/<out-exp>/<split>/：
- comprehensive_baseline.md / .json
- curve_tree.csv / curve_summary.csv
- fig_kpi_pruning_curve.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))
from config import CFG  # noqa: E402


POLICIES = ["dt", "expert", "bc", "bcq", "cql", "iql", "greedy", "random"]
LEARNED = {"dt", "bc", "bcq", "cql", "iql"}
KPI_KEYS = ["hierarchy", "crowding", "balance", "light", "potential", "stability"]
CN = {
    "hierarchy": "层次", "crowding": "拥挤", "balance": "平衡",
    "light": "光照", "potential": "潜力", "stability": "稳定",
}
DEFECT_TAGS = {
    "cross_inferior", "cross_other", "inward", "vigorous", "drooping", "weak",
}
PROTECT_TAGS = {"protect_medium_short", "protect_sturdy"}
GRID = np.array([0.10, 0.15, 0.20, 0.25, 0.30])


def read_result(directory: str, tag: str) -> dict[str, dict]:
    csv_path = os.path.join(directory, f"eval_{tag}.csv")
    actions_path = os.path.join(directory, f"actions_{tag}.json")
    if not os.path.exists(csv_path) or not os.path.exists(actions_path):
        raise SystemExit(f"缺少结果：{csv_path} 或 {actions_path}")
    with open(actions_path, encoding="utf-8") as f:
        traces = json.load(f)
    rows = {}
    with open(csv_path, encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            rows[raw["tree"]] = {"raw": raw, "trace": traces[raw["tree"]]}
    return rows


def kpi_sum(delta: dict[str, float]) -> float:
    cfg = CFG.reward
    return float(sum(cfg.kpi_weights[k] * delta[k] / cfg.kpi_scales[k] for k in KPI_KEYS))


def endpoint_metrics(item: dict, policy: str) -> dict[str, float]:
    raw, trace = item["raw"], item["trace"]
    delta = {k: float(raw[f"{k}_delta"]) for k in KPI_KEYS}
    points = sum(float(step["points_removed"]) for step in trace)
    volume_units = points / CFG.reward.rq_volume_ref
    rq_raw = sum(float(step["rq"]) for step in trace)
    ratio = float(raw["final_pruning_ratio"])
    tags = [set(step.get("tags", [])) for step in trace]
    stop_reason = raw.get("stop_reason")
    if not stop_reason and policy in LEARNED:
        stop_reason = "constraint" if ratio >= CFG.traj.max_pruning_ratio else "model"
    out = {
        **delta,
        "kpi_sum": kpi_sum(delta),
        "return": float(raw["total_reward"]),
        "ratio": ratio,
        "cuts": float(len(trace)),
        "rq": CFG.reward.alpha * rq_raw,
        "ri": CFG.reward.beta * sum(float(step["ri"]) for step in trace),
        "rc": CFG.reward.gamma * sum(float(step["rc"]) for step in trace),
        "quality_density": rq_raw / volume_units if volume_units > 0 else 0.0,
        "defect_action_rate": np.mean([bool(t & DEFECT_TAGS) for t in tags]) if tags else 0.0,
        "protect_action_rate": np.mean([bool(t & PROTECT_TAGS) for t in tags]) if tags else 0.0,
        "constraint_rate": np.mean([float(step["rc"]) < 0 for step in trace]) if trace else 0.0,
        "main_cuts": float(sum("main_branch" in t for t in tags)),
        "points_per_cut": points / len(trace) if trace else 0.0,
        "branches_per_cut": (
            np.mean([len(step.get("removed", [])) for step in trace]) if trace else 0.0
        ),
        "kpi_efficiency": kpi_sum(delta) / ratio if ratio > 0 else 0.0,
        "ideal_stop": float(0.18 <= ratio <= 0.32),   # 无区分力（随机亦约 90%），仅留 JSON
        "under_025": float(ratio < 0.25),
        "hard_stop": float(ratio >= CFG.traj.max_pruning_ratio),
        "autonomous_stop": float(stop_reason == "model") if policy in LEARNED else np.nan,
    }
    return out


def curve_points(item: dict) -> dict[float, dict[str, float]]:
    raw, trace = item["raw"], item["trace"]
    ratios = [0.0]
    values = {k: [float(raw[f"{k}_before"])] for k in KPI_KEYS}
    for step in trace:
        if "pruning_ratio_after" not in step or "kpi_after" not in step:
            raise SystemExit("过程 trace 缺 pruning_ratio_after/kpi_after，请用新版 evaluate.py 重跑")
        ratios.append(float(step["pruning_ratio_after"]))
        for k in KPI_KEYS:
            values[k].append(float(step["kpi_after"][k]))
    order = np.argsort(ratios)
    x = np.asarray(ratios)[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    before = {k: float(raw[f"{k}_before"]) for k in KPI_KEYS}
    out = {}
    for threshold in GRID:
        if unique_x[-1] + 1e-9 < threshold:
            raise SystemExit(
                f"{raw['tree']} 的过程只到 {unique_x[-1]:.3f}，未达到 {threshold:.2f}"
            )
        delta = {}
        for k in KPI_KEYS:
            y = np.asarray(values[k])[order][unique_idx]
            delta[k] = float(np.interp(threshold, unique_x, y) - before[k])
        out[float(threshold)] = {**delta, "kpi_sum": kpi_sum(delta)}
    return out


def curve_kpi_at(item: dict, x: float) -> float:
    """策略屏蔽 STOP 的过程曲线在剪枝比 x 处的 KPI 综合（线性插值，含 0 点）。"""
    raw, trace = item["raw"], item["trace"]
    ratios = [0.0]
    values = {k: [float(raw[f"{k}_before"])] for k in KPI_KEYS}
    for step in trace:
        ratios.append(float(step["pruning_ratio_after"]))
        for k in KPI_KEYS:
            values[k].append(float(step["kpi_after"][k]))
    order = np.argsort(ratios)
    ux, idx = np.unique(np.asarray(ratios)[order], return_index=True)
    if x > ux[-1] + 0.02:
        raise SystemExit(f"{raw['tree']} 过程只到 {ux[-1]:.3f}，不足以插值到 {x:.3f}")
    delta = {k: float(np.interp(x, ux, np.asarray(values[k])[order][idx]) - float(raw[f"{k}_before"]))
             for k in KPI_KEYS}
    return kpi_sum(delta)


def paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    d = a - b
    nz = d[d != 0]
    test = wilcoxon(a, b) if len(nz) >= 8 else None
    if len(nz):
        ranks = rankdata(np.abs(nz))
        effect = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    else:
        effect = 0.0
    boot = rng.choice(d, size=(5000, len(d))).mean(axis=1)
    return {
        "diff": float(d.mean()),
        "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "win": float(np.mean(d > 0)),
        "p": float(test.pvalue) if test else None,
        "effect": effect,
    }


def holm_adjust(tests: dict[str, dict]) -> None:
    valid = sorted(
        ((name, result["p"]) for name, result in tests.items() if result["p"] is not None),
        key=lambda pair: pair[1],
    )
    running = 0.0
    m = len(valid)
    for rank, (name, p) in enumerate(valid):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        tests[name]["p_holm"] = running
    for name, result in tests.items():
        if result["p"] is None:
            tests[name]["p_holm"] = None


def fmt_p(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3g}"


def safe_mean(values: np.ndarray) -> float | None:
    return None if np.all(np.isnan(values)) else float(np.nanmean(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="dev", choices=["dev", "paper"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--endpoint-exp", default="e3_v2")
    ap.add_argument("--stop-exp", default="stop_diag")
    ap.add_argument("--out-exp", default="e3_comprehensive")
    ap.add_argument("--dt-tag", default="dt_bias_m06")
    args = ap.parse_args()

    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    endpoint_dir = os.path.join(results_root, args.stage, args.endpoint_exp, args.split)
    stop_dir = os.path.join(results_root, args.stage, args.stop_exp, args.split)
    out_dir = os.path.join(results_root, args.stage, args.out_exp, args.split)
    os.makedirs(out_dir, exist_ok=True)

    endpoint = {}
    for policy in POLICIES:
        directory, tag = (stop_dir, args.dt_tag) if policy == "dt" else (endpoint_dir, policy)
        endpoint[policy] = read_result(directory, tag)
    curves = {
        policy: read_result(out_dir, f"curve_{policy}")
        for policy in POLICIES
    }
    trees = sorted(set.intersection(
        *(set(rows) for rows in [*endpoint.values(), *curves.values()])
    ))
    if not trees:
        raise SystemExit("终点与过程结果没有共同树")

    metrics = {
        policy: {
            tree: endpoint_metrics(endpoint[policy][tree], policy)
            for tree in trees
        }
        for policy in POLICIES
    }
    curve_data = {
        policy: {tree: curve_points(curves[policy][tree]) for tree in trees}
        for policy in POLICIES
    }
    for policy in POLICIES:
        for tree in trees:
            endpoint_m = metrics[policy][tree]
            at_030 = curve_data[policy][tree][0.30]
            if endpoint_m["ratio"] >= 0.30:
                endpoint_m["stop_regret_kpi"] = 0.0
                endpoint_m["stop_regret_return"] = 0.0
            else:
                endpoint_m["stop_regret_kpi"] = max(
                    0.0, at_030["kpi_sum"] - endpoint_m["kpi_sum"]
                )
                endpoint_m["stop_regret_return"] = max(
                    0.0,
                    float(curves[policy][tree]["raw"]["total_reward"])
                    - endpoint_m["return"],
                )
            x = np.r_[0.0, GRID]
            y = np.r_[0.0, [curve_data[policy][tree][float(q)]["kpi_sum"] for q in GRID]]
            endpoint_m["curve_auc"] = float(np.trapz(y, x) / GRID[-1])
            # 同剪枝量：每棵树上所有策略（含 DT）都取过程曲线在 min(DT 停止剪枝比, 0.30) 处的 KPI，
            # 统一插值口径；上限 0.30 即硬上限，避免对曲线终点之外外推
            endpoint_m["kpi_matched"] = curve_kpi_at(
                curves[policy][tree], min(metrics["dt"][tree]["ratio"], float(GRID[-1])))
            for q in GRID:
                endpoint_m[f"kpi_at_{int(q * 100):02d}"] = curve_data[policy][tree][float(q)]["kpi_sum"]

    metric_names = list(next(iter(next(iter(metrics.values())).values())).keys())
    arrays = {
        policy: {
            key: np.array([metrics[policy][tree][key] for tree in trees], dtype=float)
            for key in metric_names
        }
        for policy in POLICIES
    }
    means = {
        policy: {key: safe_mean(values) for key, values in policy_arrays.items()}
        for policy, policy_arrays in arrays.items()
    }

    compare_metrics = [
        *KPI_KEYS, "kpi_sum", "return", "ratio", "rq", "ri", "rc",
        "quality_density", "defect_action_rate", "protect_action_rate",
        "main_cuts", "points_per_cut", "branches_per_cut", "kpi_efficiency",
        "curve_auc", "kpi_matched", "stop_regret_kpi", "stop_regret_return",
    ]
    rng = np.random.default_rng(0)
    tests = {}
    for key in compare_metrics:
        per_opponent = {}
        for policy in POLICIES:
            if policy == "dt" or (key in {"return", "rq", "ri", "rc"} and policy not in LEARNED):
                continue
            per_opponent[policy] = paired(arrays["dt"][key], arrays[policy][key], rng)
        holm_adjust(per_opponent)
        tests[key] = per_opponent

    curve_rows = []
    for policy in POLICIES:
        for tree in trees:
            for q in GRID:
                curve_rows.append({
                    "tree": tree, "policy": policy, "pruning_ratio": float(q),
                    **curve_data[policy][tree][float(q)],
                })
    with open(os.path.join(out_dir, "curve_tree.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    summary_rows = []
    for policy in POLICIES:
        for q in GRID:
            selected = [r for r in curve_rows if r["policy"] == policy and r["pruning_ratio"] == q]
            summary_rows.append({
                "policy": policy,
                "pruning_ratio": float(q),
                **{key: float(np.mean([r[key] for r in selected]))
                   for key in [*KPI_KEYS, "kpi_sum"]},
            })
    with open(os.path.join(out_dir, "curve_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    def test_rows(items):
        rows = [
            "| 指标 | 对手 | 均值差(DT-对手) | 95% CI | Holm p | 效应量 | 胜率 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for key, label, digits in items:
            for policy, result in tests[key].items():
                rows.append(
                    f"| {label} | {policy} | {result['diff']:+.{digits}f} | "
                    f"[{result['ci'][0]:+.{digits}f}, {result['ci'][1]:+.{digits}f}] | "
                    f"{fmt_p(result['p_holm'])} | {result['effect']:+.2f} | "
                    f"{result['win']:.0%} |"
                )
        return rows

    lines = [
        f"# E3 多维综合基线（{args.stage}/{args.out_exp}/{args.split}，{len(trees)} 棵）",
        "",
        "主表为**同剪枝量比较**与**曲线 AUC**：各策略剪枝量不同（见第 2 节），终点 KPI 混杂“剪多少”，"
        "故只作补充。过程曲线屏蔽学习策略 STOP 至 0.30，评价选枝能力。"
        f"DT 终点口径 `{args.dt_tag}`（推理时 STOP logit 偏置，验证树上选定，测试前固定）。",
        "",
        "## 1. 主表：同剪枝量 KPI 与曲线 AUC",
        "",
        "同剪枝量 KPI：每棵树上取各策略（含 DT）过程曲线在 min(DT 停止剪枝比, 0.30) 处的 KPI 综合（线性插值）。",
        "",
        "| 策略 | 同剪枝量KPI | AUC | KPI@0.10 | KPI@0.15 | KPI@0.20 | KPI@0.25 | KPI@0.30 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        m = means[policy]
        lines.append(
            f"| {policy} | {m['kpi_matched']:.1f} | {m['curve_auc']:.1f} | "
            + " | ".join(f"{m[f'kpi_at_{int(q * 100):02d}']:.1f}" for q in GRID)
            + " |"
        )
    lines += [
        "",
        *test_rows([("kpi_matched", "同剪枝量KPI", 1), ("curve_auc", "曲线AUC", 1)]),
        "",
        "## 2. 补充：终点结果（各策略剪枝量不同）",
        "",
        "| 策略 | KPI综合 | 剪枝比 | 刀数 |",
        "|---|---:|---:|---:|",
    ]
    for policy in POLICIES:
        m = means[policy]
        lines.append(f"| {policy} | {m['kpi_sum']:.1f} | {m['ratio']:.3f} | {m['cuts']:.1f} |")
    lines += [
        "",
        "六项 KPI 变化：",
        "",
        "| 策略 | " + " | ".join(CN[k] for k in KPI_KEYS) + " |",
        "|---|" + "---:|" * len(KPI_KEYS),
    ]
    for policy in POLICIES:
        m = means[policy]
        lines.append(f"| {policy} | " + " | ".join(f"{m[k]:+.4f}" for k in KPI_KEYS) + " |")
    lines += ["", *test_rows([("kpi_sum", "终点KPI", 1)])]

    lines += [
        "",
        "## 3. 回报、选枝行为与结构安全",
        "",
        "回报只在学习方法之间比较；单位剪除量规则质量按规则分计算，专家和 BC 按规则选枝，与之同源。",
        "",
        "| 策略 | 回报 | αRq | βRi | γRc | 单位剪除量规则质量 | 约束触发 | 骨干枝/树 | 点/刀 | 分枝/刀 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        m = means[policy]
        lines.append(
            f"| {policy} | {m['return']:.0f} | {m['rq']:.0f} | {m['ri']:.0f} | "
            f"{m['rc']:.1f} | {m['quality_density']:.2f} | "
            f"{m['constraint_rate']:.0%} | {m['main_cuts']:.2f} | {m['points_per_cut']:.1f} | "
            f"{m['branches_per_cut']:.2f} |"
        )
    lines += [
        "",
        *test_rows([("return", "回报", 0), ("quality_density", "单位剪除量规则质量", 2),
                    ("main_cuts", "骨干枝/树", 2)]),
        "",
        "不良枝动作占比在学习方法中达到 99%-100%，缺乏区分度，保留在 JSON 中。",
        "",
        "## 4. 停止行为（仅描述，不做优劣检验）",
        "",
        "训练数据的停止档按树序号轮转、与树状态无关，评估时专家等策略也不自然停止，"
        "没有可对照的“应停位置”。自主 STOP：学习策略自己选择停止；触顶：剪枝比达到硬上限 0.30。",
        "",
        "| 策略 | 平均剪枝比 | 剪枝比<0.25占比 | 自主STOP | 触顶 |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        m = means[policy]
        auto = "—" if policy not in LEARNED else f"{m['autonomous_stop']:.0%}"
        lines.append(
            f"| {policy} | {m['ratio']:.3f} | {m['under_025']:.0%} | {auto} | {m['hard_stop']:.0%} |"
        )

    report = {
        "trees": trees,
        "dt_tag": args.dt_tag,
        "grid": GRID.tolist(),
        "means": means,
        "tests_dt_vs_others": tests,
    }
    with open(os.path.join(out_dir, "comprehensive_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "comprehensive_baseline.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for policy in POLICIES:
            y = [next(r["kpi_sum"] for r in summary_rows
                      if r["policy"] == policy and r["pruning_ratio"] == q) for q in GRID]
            ax.plot(GRID, y, marker="o", linewidth=2 if policy == "dt" else 1.3,
                    label=policy.upper() if policy in LEARNED else policy.title())
        ax.set_xlabel("Pruning ratio")
        ax.set_ylabel("Composite KPI improvement")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_kpi_pruning_curve.png"), dpi=220)
        plt.close(fig)

        forest_metrics = [
            ("kpi_matched", "Matched-amount KPI"),
            ("curve_auc", "Curve AUC"),
            ("kpi_sum", "Final KPI (supplement)"),
            ("quality_density", "Rule-quality density"),
            ("main_cuts", "Main-branch cuts"),
        ]
        opponents = [p for p in POLICIES if p != "dt"]
        fig, axes = plt.subplots(1, len(forest_metrics), figsize=(15.5, 4.8), sharey=True)
        for ax, (key, title) in zip(axes, forest_metrics):
            effects = [tests[key][p]["effect"] for p in opponents]
            y = np.arange(len(opponents))
            ax.axvline(0, color="0.45", linewidth=1)
            ax.scatter(effects, y, color="#1f5a75", s=28)
            ax.set_xlim(-1.05, 1.05)
            ax.set_title(title, fontsize=9)
            ax.grid(axis="x", alpha=0.2)
            ax.set_yticks(y)
            if ax is axes[0]:
                ax.set_yticklabels(opponents)
            else:
                ax.tick_params(axis="y", labelleft=False)
        axes[0].invert_yaxis()
        fig.supxlabel("Paired rank-biserial effect (DT - baseline)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_effect_sizes.png"), dpi=220)
        plt.close(fig)
    except ImportError:
        print("未安装 matplotlib，跳过曲线 PNG；CSV 与分析结果不受影响。")

    print("\n".join(lines))
    print("\n已保存:", out_dir)


if __name__ == "__main__":
    main()
