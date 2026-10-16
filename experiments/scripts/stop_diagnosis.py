"""汇总 DT STOP 诊断结果。

读取同一实验目录中的 eval_<tag>.csv，对不同目标回报、STOP 偏置和最低剪枝比
设置做逐树配对比较。该脚本只分析验证结果，不运行模型。

用法：
    python experiments/scripts/stop_diagnosis.py --exp stop_diag --ref dt_base
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))
from config import CFG  # noqa: E402


KPI_KEYS = ["hierarchy", "crowding", "balance", "light", "potential", "stability"]


def load(path: str) -> dict[str, dict]:
    cfg = CFG.reward
    rows = {}
    with open(path, encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = {k: v for k, v in raw.items()}
            row["reward"] = float(raw["total_reward"])
            row["ratio"] = float(raw["final_pruning_ratio"])
            row["steps"] = float(raw["steps"])
            row["kpi_sum"] = sum(
                cfg.kpi_weights[k] * float(raw[f"{k}_delta"]) / cfg.kpi_scales[k]
                for k in KPI_KEYS
            )
            row["suppressed"] = float(raw.get("suppressed_stop_steps") or 0)
            rows[raw["tree"]] = row
    return rows


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    test = wilcoxon(a, b) if len(a) >= 8 and np.any(diff != 0) else None
    return {
        "diff": float(diff.mean()),
        "win": float(np.mean(diff > 0)),
        "p": float(test.pvalue) if test else None,
    }


def fmt_test(test: dict, digits: int) -> str:
    p = test["p"]
    p_text = "NA" if p is None else f"{p:.3g}"
    return f"{test['diff']:+.{digits}f} (p={p_text}, 胜率={test['win']:.0%})"


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="stop_diag")
    ap.add_argument("--split", default="val")
    ap.add_argument("--stage", default="dev")
    ap.add_argument("--ref", default="dt_base")
    args = ap.parse_args()

    directory = os.path.join(os.path.dirname(__file__), "..", "results",
                             args.stage, args.exp, args.split)
    files = sorted(
        name for name in os.listdir(directory)
        if name.startswith("eval_") and name.endswith(".csv")
    )
    data = {name[5:-4]: load(os.path.join(directory, name)) for name in files}
    if args.ref not in data:
        raise SystemExit(f"缺少参考结果 eval_{args.ref}.csv")

    common = sorted(set.intersection(*(set(rows) for rows in data.values())))
    if not common:
        raise SystemExit("各结果文件没有共同树")

    arrays = {
        tag: {
            key: np.array([rows[tree][key] for tree in common], dtype=float)
            for key in ("reward", "kpi_sum", "ratio", "steps", "suppressed")
        }
        for tag, rows in data.items()
    }
    ref = arrays[args.ref]
    tests = {
        tag: {key: paired(values[key], ref[key]) for key in ("reward", "kpi_sum", "ratio")}
        for tag, values in arrays.items() if tag != args.ref
    }

    means = {}
    lines = [
        f"# DT STOP 诊断（{args.stage}/{args.exp}/{args.split}，{len(common)} 棵）",
        "",
        "## 汇总",
        "",
        "| 设置 | 回报 | KPI综合 | 剪枝比 | 刀数 | 模型自主STOP | 理想区间 | <0.25 | 屏蔽STOP次数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, values in arrays.items():
        rows = data[tag]
        model_stop = np.mean([rows[t].get("stop_reason") == "model" for t in common])
        ideal = np.mean((values["ratio"] >= 0.18) & (values["ratio"] <= 0.32))
        below = np.mean(values["ratio"] < 0.25)
        means[tag] = {
            **{key: float(value.mean()) for key, value in values.items()},
            "model_stop_rate": float(model_stop),
            "ideal_rate": float(ideal),
            "below_025_rate": float(below),
            "corr_ratio_reward": safe_corr(values["ratio"], values["reward"]),
            "corr_ratio_kpi": safe_corr(values["ratio"], values["kpi_sum"]),
        }
        lines.append(
            f"| {tag} | {values['reward'].mean():.0f} | {values['kpi_sum'].mean():.1f} | "
            f"{values['ratio'].mean():.3f} | {values['steps'].mean():.1f} | "
            f"{model_stop:.0%} | {ideal:.0%} | {below:.0%} | {values['suppressed'].mean():.1f} |"
        )

    lines += [
        "",
        f"## 相对 {args.ref} 的逐树配对差值",
        "",
        "| 设置 | 回报差 | KPI综合差 | 剪枝比差 |",
        "|---|---|---|---|",
    ]
    for tag, result in tests.items():
        lines.append(
            f"| {tag} | {fmt_test(result['reward'], 0)} | "
            f"{fmt_test(result['kpi_sum'], 1)} | {fmt_test(result['ratio'], 3)} |"
        )

    lines += [
        "",
        "## 解释规则",
        "",
        "- 提高最低剪枝比后，回报和 KPI 同时追平：主要瓶颈是提前 STOP。",
        "- 剪枝比提高但回报/KPI仍落后：选枝质量同样存在问题。",
        "- 调目标回报即可改善：优先采用目标回报校准，无需重训。",
        "- 只有屏蔽 STOP 才能改善：需进一步考虑 STOP 校准或独立停止头。",
    ]

    report = {
        "trees": common,
        "ref": args.ref,
        "means": means,
        "tests_vs_ref": tests,
    }
    with open(os.path.join(directory, "stop_diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    with open(os.path.join(directory, "stop_diagnosis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
