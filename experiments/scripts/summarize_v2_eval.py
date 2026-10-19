"""汇总 v2 在线评估 CSV 的停止、回报和六维 KPI。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

KPI = ("hierarchy", "crowding", "balance", "light", "potential", "stability")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    args = ap.parse_args()
    output = {}
    for value in args.csv:
        path = Path(value)
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ratio = np.asarray([float(r["final_pruning_ratio"]) for r in rows])
        reward = np.asarray([float(r["total_reward"]) for r in rows])
        steps = np.asarray([int(r["steps"]) for r in rows])
        reasons = [r["stop_reason"] for r in rows]
        output[path.stem] = {
            "n": len(rows),
            "model_stop": int(sum(x == "model" for x in reasons)),
            "model_stop_rate": float(np.mean([x == "model" for x in reasons])),
            "ratio_mean": float(ratio.mean()),
            "ratio_median": float(np.median(ratio)),
            "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()),
            "reward_mean": float(reward.mean()),
            "reward_median": float(np.median(reward)),
            "steps_mean": float(steps.mean()),
            "reward_per_step": float(reward.sum() / steps.sum()),
            "kpi_delta_mean": {
                k: float(np.mean([float(r[f"{k}_delta"]) for r in rows])) for k in KPI},
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
