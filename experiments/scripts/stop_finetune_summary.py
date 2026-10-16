"""汇总 STOP 头微调候选，并按预先固定标准推荐验证集候选。

标准：
1. 理想区间命中率 >= 90%
2. 自主 STOP 比例 >= 50%
3. 分枝部分完全冻结，因此沿用当前 DT 过程 AUC
4. 满足以上条件者中，选择“未实现 KPI@0.30”最小者
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))
from config import CFG  # noqa: E402


KPI_KEYS = ["hierarchy", "crowding", "balance", "light", "potential", "stability"]


def load_eval(path: str) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            delta = {k: float(raw[f"{k}_delta"]) for k in KPI_KEYS}
            out[raw["tree"]] = {
                "reward": float(raw["total_reward"]),
                "ratio": float(raw["final_pruning_ratio"]),
                "kpi": float(sum(
                    CFG.reward.kpi_weights[k] * delta[k] / CFG.reward.kpi_scales[k]
                    for k in KPI_KEYS
                )),
                "stop_reason": raw.get("stop_reason") or (
                    "constraint" if float(raw["final_pruning_ratio"]) >= 0.30 else "model"
                ),
            }
    return out


def load_curve_030(path: str) -> dict[str, float]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["policy"] == "dt" and abs(float(row["pruning_ratio"]) - 0.30) < 1e-9:
                out[row["tree"]] = float(row["kpi_sum"])
    return out


def summarize(rows: dict[str, dict], curve: dict[str, float]) -> dict:
    trees = sorted(set(rows) & set(curve))
    ratio = np.array([rows[t]["ratio"] for t in trees])
    kpi = np.array([rows[t]["kpi"] for t in trees])
    unrealized = np.array([
        0.0 if rows[t]["ratio"] >= 0.30 else max(0.0, curve[t] - rows[t]["kpi"])
        for t in trees
    ])
    return {
        "n": len(trees),
        "reward": float(np.mean([rows[t]["reward"] for t in trees])),
        "kpi": float(kpi.mean()),
        "ratio": float(ratio.mean()),
        "ideal_rate": float(np.mean((ratio >= 0.18) & (ratio <= 0.32))),
        "autonomous_rate": float(np.mean([rows[t]["stop_reason"] == "model" for t in trees])),
        "hard_rate": float(np.mean([rows[t]["stop_reason"] == "constraint" for t in trees])),
        "unrealized_kpi": float(unrealized.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="stop_finetune")
    ap.add_argument("--split", default="val")
    ap.add_argument("--stage", default="dev")
    ap.add_argument("--ideal-min", type=float, default=0.90)
    ap.add_argument("--autonomous-min", type=float, default=0.50)
    args = ap.parse_args()

    results = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    directory = os.path.join(results, args.stage, args.exp, args.split)
    stop_dir = os.path.join(results, "dev", "stop_diag", "val")
    curve_path = os.path.join(results, "dev", "e3_comprehensive", "val", "curve_tree.csv")
    curve = load_curve_030(curve_path)
    data = {
        "base": load_eval(os.path.join(stop_dir, "eval_dt_base.csv")),
        "bias_m06": load_eval(os.path.join(stop_dir, "eval_dt_bias_m06.csv")),
    }
    for name in sorted(os.listdir(directory)):
        if name.startswith("eval_dt_w") and name.endswith(".csv"):
            tag = name[5:-4]
            data[tag] = load_eval(os.path.join(directory, name))
    if len(data) <= 2:
        raise SystemExit(f"{directory} 中没有 STOP 微调评估结果")

    summary = {tag: summarize(rows, curve) for tag, rows in data.items()}
    meets = {
        tag: (
            m["ideal_rate"] >= args.ideal_min
            and m["autonomous_rate"] >= args.autonomous_min
        )
        for tag, m in summary.items()
    }
    eligible = [
        tag for tag in summary
        if tag not in {"base", "bias_m06"} and meets[tag]
    ]
    recommended = min(
        eligible, key=lambda tag: summary[tag]["unrealized_kpi"]
    ) if eligible else None
    eligible_all = [tag for tag in summary if meets[tag]]
    recommended_overall = min(
        eligible_all, key=lambda tag: summary[tag]["unrealized_kpi"]
    ) if eligible_all else None

    lines = [
        "# DT STOP 头微调候选（验证集）",
        "",
        f"筛选约束：理想区间 >= {args.ideal_min:.0%}，自主 STOP >= {args.autonomous_min:.0%}；"
        "分枝网络冻结，过程 AUC 沿用当前 DT。",
        "",
        "| 候选 | 回报 | KPI综合 | 剪枝比 | 理想区间 | 自主STOP | 硬上限 | 未实现KPI@0.30 | 合格 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, m in summary.items():
        ok = meets[tag]
        lines.append(
            f"| {tag} | {m['reward']:.0f} | {m['kpi']:.1f} | {m['ratio']:.3f} | "
            f"{m['ideal_rate']:.0%} | {m['autonomous_rate']:.0%} | "
            f"{m['hard_rate']:.0%} | {m['unrealized_kpi']:.1f} | "
            f"{'是' if ok else '否'} |"
        )
    lines += [
        "",
        f"**微调候选推荐**：{recommended or '无；当前微调未同时满足约束'}",
        "",
        f"**包含既有校准的总体推荐**：{recommended_overall or '无'}",
        "",
        "该推荐仅用于冻结验证集配置；测试集仍须在全部实验定稿后一次性运行。",
    ]
    report = {
        "criteria": {
            "ideal_min": args.ideal_min,
            "autonomous_min": args.autonomous_min,
            "auc": "branch network frozen; unchanged",
        },
        "summary": summary,
        "meets_constraints": meets,
        "eligible": eligible,
        "recommended": recommended,
        "recommended_overall": recommended_overall,
    }
    with open(os.path.join(directory, "stop_finetune_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    with open(os.path.join(directory, "stop_finetune_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
