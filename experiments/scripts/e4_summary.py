"""汇总 E4 五个核心消融，并写入验证结果目录。

主指标是 AUC、KPI@0.30、自主剪枝比和终点 KPI。
Holm 校正在每个主指标内对五个核心变体分别进行。
补充变体单独报告，不进入校正。

用法（项目根目录，两个新变体评估完成之后）：
    py -3.12 experiments/scripts/e4_summary.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))
import e3_comprehensive as e3  # noqa: E402


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "dev"))
ABL = os.path.join(ROOT, "e4_ablation", "val")
CORE = [
    ("k1", "去时序上下文"),
    ("noknowledge", "去园艺知识"),
    ("norq", "去规则质量奖励"),
    ("nori", "去六维KPI奖励"),
    ("norc", "去安全约束奖励"),
]
SUPPLEMENT = [
    ("expert", "只用专家轨迹"),
    ("noprotect", "去功能枝保护"),
]
PRIMARY = ["AUC", "KPI@0.30", "ratio", "kpi_sum"]


def auc_of(item):
    points = e3.curve_points(item)
    x = np.r_[0.0, e3.GRID]
    y = np.r_[0.0, [points[float(q)]["kpi_sum"] for q in e3.GRID]]
    grid = {float(q): points[float(q)]["kpi_sum"] for q in e3.GRID}
    return float(np.trapezoid(y, x) / e3.GRID[-1]), grid


def collect(directory: str, tag: str, trees: list[str], endpoint: bool) -> dict[str, np.ndarray]:
    rows = e3.read_result(directory, tag)
    missing = [tree for tree in trees if tree not in rows]
    if missing:
        raise SystemExit(f"{tag} 缺少 {len(missing)} 棵树")
    if endpoint:
        metrics = {tree: e3.endpoint_metrics(rows[tree], "dt") for tree in trees}
        return {key: np.array([metrics[tree][key] for tree in trees])
                for key in ("ratio", "kpi_sum", "constraint_rate", "autonomous_stop")}
    auc, grids = [], {float(q): [] for q in e3.GRID}
    for tree in trees:
        value, grid = auc_of(rows[tree])
        auc.append(value)
        for q, item in grid.items():
            grids[q].append(item)
    out = {"AUC": np.array(auc)}
    out.update({f"KPI@{q:.2f}": np.array(values) for q, values in grids.items()})
    return out


def record(tests: dict, name: str, observed: np.ndarray, reference: np.ndarray,
           rng: np.random.Generator, corrected: bool) -> None:
    stat = e3.paired(observed, reference, rng)
    stat.update(mean=float(observed.mean()), corrected=corrected, p_holm=None)
    tests[name] = stat


def fmt(value, digits):
    return "NA" if value is None else f"{value:.{digits}f}"


def markdown_table(labels: dict[str, str], tests: dict, digits: dict) -> list[str]:
    lines = ["| 变体 | 指标 | 均值 | 相对完整DT | 95% CI | p | Holm p | 效应量 |",
             "|---|---|---:|---:|---|---:|---:|---:|"]
    for name, label in labels.items():
        for metric, stat in tests[name].items():
            ci = f"{stat['ci'][0]:.3g}, {stat['ci'][1]:.3g}"
            lines.append(
                f"| {label} | {metric} | {fmt(stat['mean'], digits[metric])} "
                f"| {fmt(stat['diff'], digits[metric])} | {ci} | {e3.fmt_p(stat['p'])} "
                f"| {e3.fmt_p(stat['p_holm'])} | {stat['effect']:+.2f} |"
            )
    return lines


def main():
    required = [tag for tag, _ in CORE] + [f"curve_{tag}" for tag, _ in CORE]
    missing = [tag for tag in required
               if not os.path.exists(os.path.join(ABL, f"eval_{tag}.csv"))
               or not os.path.exists(os.path.join(ABL, f"actions_{tag}.json"))]
    if missing:
        raise SystemExit("核心结果未齐，缺少：" + ", ".join(missing))

    rng = np.random.default_rng(0)
    ref_end = e3.read_result(os.path.join(ROOT, "stop_diag", "val"), "dt_base")
    ref_curve = e3.read_result(os.path.join(ROOT, "e3_comprehensive", "val"), "curve_dt")
    trees = sorted(set(ref_end) & set(ref_curve))
    reference = {**collect(os.path.join(ROOT, "stop_diag", "val"), "dt_base", trees, True),
                 **collect(os.path.join(ROOT, "e3_comprehensive", "val"), "curve_dt", trees, False)}

    labels = dict(CORE)
    variants = {}
    for tag, _ in CORE:
        variants[tag] = {**collect(ABL, tag, trees, True), **collect(ABL, "curve_" + tag, trees, False)}
    present_supplement = []
    for tag, label in SUPPLEMENT:
        paths = [os.path.join(ABL, f"eval_{tag}.csv"), os.path.join(ABL, f"actions_{tag}.json"),
                 os.path.join(ABL, f"eval_curve_{tag}.csv"), os.path.join(ABL, f"actions_curve_{tag}.json")]
        if all(os.path.exists(path) for path in paths):
            labels[tag] = label
            variants[tag] = {**collect(ABL, tag, trees, True),
                             **collect(ABL, "curve_" + tag, trees, False)}
            present_supplement.append(tag)

    tests = {tag: {} for tag in variants}
    for metric in reference:
        group = {tag: {} for tag in variants}
        for tag in variants:
            record(group, tag, variants[tag][metric], reference[metric], rng, tag in dict(CORE))
        core_tests = {tag: group[tag] for tag, _ in CORE}
        if metric in PRIMARY:
            e3.holm_adjust(core_tests)
        for tag in variants:
            tests[tag][metric] = group[tag]

    payload = {
        "n_trees": len(trees),
        "reference": {metric: float(values.mean()) for metric, values in reference.items()},
        "primary_metrics": PRIMARY,
        "holm_family": "每个主指标内的五个核心变体",
        "variants": tests,
    }
    os.makedirs(ABL, exist_ok=True)
    with open(os.path.join(ABL, "e4_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ABL, "e4_ablation_tests.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["group", "variant", "metric", "mean", "diff", "ci_low", "ci_high",
                  "p", "p_holm", "effect"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for tag, label in labels.items():
            group = "core" if tag in dict(CORE) else "supplement"
            for metric, stat in tests[tag].items():
                writer.writerow({
                    "group": group, "variant": label, "metric": metric,
                    "mean": stat["mean"], "diff": stat["diff"],
                    "ci_low": stat["ci"][0], "ci_high": stat["ci"][1],
                    "p": stat["p"], "p_holm": stat["p_holm"], "effect": stat["effect"],
                })

    digits = {metric: (3 if metric in ("ratio", "constraint_rate", "autonomous_stop") else 1)
              for metric in reference}
    lines = [
        "# E4 核心消融（验证集）",
        "",
        f"完整 DT 参照：{len(trees)} 棵验证树；终点为 bias 0，过程曲线为剪枝比 0.30 前屏蔽 STOP。",
        "",
        "| 参照指标 | 均值 |",
        "|---|---:|",
    ]
    for metric, values in reference.items():
        lines.append(f"| {metric} | {fmt(float(values.mean()), digits[metric])} |")
    lines += ["", "## 核心变体", ""]
    lines += markdown_table(dict(CORE), tests, digits)
    lines += [
        "",
        "Holm 校正只作用于 AUC、KPI@0.30、ratio、kpi_sum 四个主指标，且每个指标内校正五个核心比较。",
        "",
        "## 补充变体",
        "",
    ]
    if present_supplement:
        lines += markdown_table({tag: labels[tag] for tag in present_supplement}, tests, digits)
    else:
        lines.append("无可用补充结果。")
    lines += [
        "",
        "去功能枝保护在标准验证 rollout 上的保护动作率为 0，其不显著结果不能解释为保护模块无贡献。",
        "",
    ]
    text = "\n".join(lines)
    with open(os.path.join(ABL, "e4_ablation.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("\n已写入", ABL)


if __name__ == "__main__":
    main()
