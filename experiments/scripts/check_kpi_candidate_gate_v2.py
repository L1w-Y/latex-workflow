"""候选级KPI修复的预设门槛；失败时不扩展其余E4消融。"""
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
OUT = EXP / "results/dev/kpi_candidate_gate_v2/model_val"
REPORT_ROOT = EXP / "results/dev/kpi_candidate_v2"
SCALE_PATH = EXP / "data/graph_cache_v2_curated/kpi_scales.json"
CANDIDATE_ROOT = "experiments/data/candidate_kpi_v2_curated"
KPI = ("hierarchy", "crowding", "balance", "light", "potential", "stability")
WEIGHTS = {
    "hierarchy": 5.0, "crowding": 10.0, "balance": 8.0,
    "light": 8.0, "potential": 2.0, "stability": 5.0,
}
MODELS = {
    "full": "checkpoints_dt_kpicandidate_v2/dt_best.pt",
    "no_kpi": "checkpoints_dt_kpicandidate_no_kpi_v2/dt_best.pt",
}


def score(delta: dict, scales: dict) -> float:
    return sum(WEIGHTS[k] * delta[k] / scales[k] for k in KPI)


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_diag = REPORT_ROOT / "candidate_diagnosis.json"
    subprocess.run([
        sys.executable, "experiments/scripts/diagnose_candidate_kpi_v2.py",
        "--checkpoint", f"experiments/{MODELS['full']}",
        "--candidate-root", CANDIDATE_ROOT,
        "--out", str(candidate_diag.relative_to(ROOT)),
    ], cwd=ROOT, check=True)

    for tag, checkpoint in MODELS.items():
        output = OUT / f"eval_{tag}_curve.csv"
        if output.exists():
            print(f"[{tag}] 选模曲线已存在，跳过", flush=True)
            continue
        subprocess.run([
            sys.executable, "evaluate_v2.py", "--checkpoint", checkpoint,
            "--split", "model_val", "--exp", "kpi_candidate_gate_v2",
            "--tag", f"{tag}_curve", "--diagnose-stop-continuation",
        ], cwd=DT, check=True)

    policy_diag = REPORT_ROOT / "policy_diagnosis.json"
    subprocess.run([
        sys.executable, "experiments/scripts/diagnose_kpi_policy_v2.py",
        "--checkpoint", f"experiments/{MODELS['full']}",
        "--result-dir",
        str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "--out", str(policy_diag.relative_to(ROOT)).replace("\\", "/"),
    ], cwd=ROOT, check=True)

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
            x = [0.0] + [
                float(s["pruning_ratio_after"]) for s in traces[tree]]
            y = [0.0] + [score({
                k: float(s["kpi_after"][k]) - float(row[f"{k}_before"])
                for k in KPI
            }, scales) for s in traces[tree]]
            auc.append(float(np.trapz(y, x) / 0.32))
        metrics[tag] = {
            "kpi_cap": float(np.mean(endpoint)),
            "kpi_auc": float(np.mean(auc)),
        }

    with candidate_diag.open(encoding="utf-8") as f:
        candidate = json.load(f)
    with policy_diag.open(encoding="utf-8") as f:
        policy = json.load(f)
    mean_dimension_improvement = float(np.mean([
        x["mae_improvement"] for x in candidate["dimensions"].values()
    ]))
    branch_changed = policy["branch_kpi_counterfactual"][
        "argmax_changed_fraction_when_kpi_zeroed"]
    checks = {
        "candidate_mean_mae_improvement>=0.10":
            mean_dimension_improvement >= 0.10,
        "candidate_utility_state_spearman>=0.20":
            candidate["candidate_utility_mean_state_spearman"] >= 0.20,
        "candidate_top1_lift_over_random>=0.10":
            candidate["candidate_utility_top1_lift"] >= 0.10,
        "branch_argmax_changed_when_kpi_zeroed>=0.10":
            branch_changed >= 0.10,
        "full_kpi_cap>=no_kpi":
            metrics["full"]["kpi_cap"] >= metrics["no_kpi"]["kpi_cap"],
        "full_kpi_auc>=no_kpi":
            metrics["full"]["kpi_auc"] >= metrics["no_kpi"]["kpi_auc"],
    }
    report = {
        "candidate_diagnosis": candidate,
        "policy_diagnosis": policy,
        "model_val": metrics,
        "derived": {
            "candidate_mean_mae_improvement": mean_dimension_improvement,
            "branch_argmax_changed_fraction": branch_changed,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    path = REPORT_ROOT / "gate.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidate_mean_mae_improvement": mean_dimension_improvement,
        "candidate_state_spearman":
            candidate["candidate_utility_mean_state_spearman"],
        "candidate_top1_lift": candidate["candidate_utility_top1_lift"],
        "branch_argmax_changed_fraction": branch_changed,
        "model_val": metrics,
        "checks": checks,
        "passed": report["passed"],
    }, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("候选级KPI修复未通过预设门槛；不要扩展其余E4消融")


if __name__ == "__main__":
    main()
