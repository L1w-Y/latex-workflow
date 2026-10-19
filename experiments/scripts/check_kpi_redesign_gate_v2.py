"""在10棵选模树上验收KPI实效化；失败则停止，不扩展到完整E4。"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
DT = ROOT / "dt_pruning"
OUT = EXP / "results/dev/kpi_redesign_gate_v2/model_val"
SCALE_PATH = EXP / "data/graph_cache_v2_curated/kpi_scales.json"
KPI = ("hierarchy", "crowding", "balance", "light", "potential", "stability")
WEIGHTS = {
    "hierarchy": 5.0, "crowding": 10.0, "balance": 8.0,
    "light": 8.0, "potential": 2.0, "stability": 5.0}
MODELS = {
    "full": "checkpoints_dt_kpifix_v2/dt_best.pt",
    "no_kpi": "checkpoints_dt_kpifix_no_kpi_v2/dt_best.pt",
}


def score(delta, scales):
    return sum(WEIGHTS[k] * delta[k] / scales[k] for k in KPI)


def main():
    diagnosis = EXP / "results/dev/kpi_redesign_v2/diagnosis_after.json"
    subprocess.run([
        sys.executable, "experiments/scripts/diagnose_kpi_head_v2.py",
        "--checkpoint", MODELS["full"], "--out",
        "results/dev/kpi_redesign_v2/diagnosis_after.json",
    ], cwd=ROOT, check=True)
    for tag, checkpoint in MODELS.items():
        output = OUT / f"eval_{tag}_curve.csv"
        if output.exists():
            print(f"[{tag}] 选模曲线已存在，跳过", flush=True)
            continue
        subprocess.run([
            sys.executable, "evaluate_v2.py", "--checkpoint", checkpoint,
            "--split", "model_val", "--exp", "kpi_redesign_gate_v2",
            "--tag", f"{tag}_curve", "--diagnose-stop-continuation",
        ], cwd=DT, check=True)

    with SCALE_PATH.open(encoding="utf-8") as f:
        scales = json.load(f)["scales"]
    metrics = {}
    for tag in MODELS:
        with (OUT / f"eval_{tag}_curve.csv").open(encoding="utf-8") as f:
            rows = {r["tree"]: r for r in csv.DictReader(f)}
        with (OUT / f"actions_{tag}_curve.json").open(encoding="utf-8") as f:
            traces = json.load(f)
        endpoint, auc = [], []
        for tree, row in rows.items():
            endpoint.append(score(
                {k: float(row[f"{k}_delta"]) for k in KPI}, scales))
            x = [0.0] + [float(s["pruning_ratio_after"]) for s in traces[tree]]
            y = [0.0] + [score({
                k: float(s["kpi_after"][k]) - float(row[f"{k}_before"])
                for k in KPI}, scales) for s in traces[tree]]
            auc.append(float(np.trapz(y, x) / 0.32))
        metrics[tag] = {"kpi_cap": float(np.mean(endpoint)),
                        "kpi_auc": float(np.mean(auc))}

    with diagnosis.open(encoding="utf-8") as f:
        diag = json.load(f)
    checks = {
        "mean_mae_improvement>=0.10": diag["mean_mae_improvement"] >= 0.10,
        "mean_spearman>=0.20": diag["mean_spearman"] >= 0.20,
        "kpi_gradient_ratio>=0.01": (
            diag["shared_gradient_norms"]["weighted_kpi_over_branch"] >= 0.01),
        "full_kpi_cap>=no_kpi": (
            metrics["full"]["kpi_cap"] >= metrics["no_kpi"]["kpi_cap"]),
        "full_kpi_auc>=no_kpi": (
            metrics["full"]["kpi_auc"] >= metrics["no_kpi"]["kpi_auc"]),
    }
    report = {"diagnosis": diag, "model_val": metrics, "checks": checks,
              "passed": all(checks.values())}
    path = EXP / "results/dev/kpi_redesign_v2/gate.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "mean_mae_improvement": diag["mean_mae_improvement"],
        "mean_spearman": diag["mean_spearman"],
        "gradient_ratio": diag["shared_gradient_norms"][
            "weighted_kpi_over_branch"],
        "model_val": metrics, "checks": checks,
        "passed": report["passed"],
    }, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("KPI实效化未通过预设门槛；不要继续训练其余消融")


if __name__ == "__main__":
    main()
