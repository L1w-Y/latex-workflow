"""汇总 v2 模型 STOP 后继续至安全上限的诊断结果。"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

KPI = ("hierarchy", "crowding", "balance", "light", "potential", "stability")


def stats(values):
    x = np.asarray(list(values), dtype=float)
    return {
        "mean": float(x.mean()), "median": float(np.median(x)),
        "min": float(x.min()), "max": float(x.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--gain-threshold", type=float, default=80.0)
    args = ap.parse_args()
    with open(args.csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["model_stopped"] == "True"]
    if not rows:
        raise SystemExit("结果中没有模型自主停止样本")
    per_step = [
        float(r["continued_reward"]) / int(r["continued_steps"])
        for r in rows if int(r["continued_steps"]) > 0]
    above = [
        {
            "tree": r["tree"],
            "stop_ratio": float(r["stop_ratio"]),
            "continued_reward_per_step": (
                float(r["continued_reward"]) / int(r["continued_steps"])),
        }
        for r in rows
        if (int(r["continued_steps"]) > 0
            and float(r["continued_reward"]) / int(r["continued_steps"])
            > args.gain_threshold)
    ]
    early = [r for r in rows if float(r["stop_ratio"]) < 0.25]
    result = {
        "model_stop_trees": len(rows),
        "early_stop_below_0.25": len(early),
        "stop_ratio": stats(float(r["stop_ratio"]) for r in rows),
        "continued_steps": stats(int(r["continued_steps"]) for r in rows),
        "continued_reward": stats(float(r["continued_reward"]) for r in rows),
        "continued_reward_per_step": stats(per_step),
        "fraction_residual_per_step_above_threshold": float(
            np.mean(np.asarray(per_step) > args.gain_threshold)),
        "trees_above_threshold": above,
        "early_continued_reward_per_step": (
            stats(float(r["continued_reward"]) / int(r["continued_steps"])
                  for r in early if int(r["continued_steps"]) > 0)
            if early else None),
        "continued_kpi_delta_mean": {
            key: float(np.mean([float(r[f"{key}_continued_delta"]) for r in rows]))
            for key in KPI
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
