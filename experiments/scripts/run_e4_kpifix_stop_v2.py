"""为KPI修复后的完整模型和五个消融统一训练on-policy Stop Head。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
DT = ROOT / "dt_pruning"
MODELS = {
    "full": "checkpoints_dt_kpifix_v2/dt_best.pt",
    "no_context": "checkpoints_dt_e4_kpifix_no_context_v2/dt_best.pt",
    "no_gardening": "checkpoints_dt_e4_kpifix_no_gardening_v2/dt_best.pt",
    "no_rq": "checkpoints_dt_e4_kpifix_no_rq_v2/dt_best.pt",
    "no_kpi": "checkpoints_dt_kpifix_no_kpi_v2/dt_best.pt",
    "no_rc": "checkpoints_dt_e4_kpifix_no_rc_v2/dt_best.pt",
}


def finished(root):
    log = root / "log.json"
    if not log.exists() or not (root / "dt_stop_onpolicy_best.pt").exists():
        return False
    with log.open(encoding="utf-8") as f:
        rows = json.load(f)
    return bool(rows) and rows[-1].get("epoch") == 100


def main():
    for name, checkpoint in MODELS.items():
        if not (EXP / checkpoint).exists():
            raise SystemExit(f"缺少 {EXP / checkpoint}")
        data = f"data/stop_onpolicy_e4_kpifix_{name}_v2"
        if not (EXP / data / "meta.json").exists():
            subprocess.run([
                sys.executable, "experiments/scripts/collect_onpolicy_stop_v2.py",
                "--checkpoint", checkpoint, "--candidate-cap", "8",
                "--out", data,
            ], cwd=ROOT, check=True)
        out = f"checkpoints_dt_e4_kpifix_{name}_stop_v2"
        if finished(EXP / out):
            print(f"[{name}] 已完成，跳过", flush=True)
            continue
        subprocess.run([
            sys.executable, "train_stop_onpolicy_v2.py",
            "--checkpoint", checkpoint, "--data", data,
            "--epochs", "100", "--steps-per-epoch", "50",
            "--batch-size", "32", "--out", out,
        ], cwd=DT, check=True)


if __name__ == "__main__":
    main()
