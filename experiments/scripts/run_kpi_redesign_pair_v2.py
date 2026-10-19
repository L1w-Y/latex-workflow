"""训练KPI实效化完整模型及严格去KPI对照；通过门槛前不训练其余消融。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DT = ROOT / "dt_pruning"
EXP = ROOT / "experiments"
EPOCHS = 60

JOBS = {
    "full": [
        "--context-len", "16", "--reward-set", "full",
        "--policies", "gardening,topk,kpi",
        "--init-from", "checkpoints_bc_v2/bc_best.pt",
        "--out", "checkpoints_dt_kpifix_v2"],
    "no_kpi": [
        "--context-len", "16", "--reward-set", "nori",
        "--policies", "gardening,topk",
        "--init-from", "checkpoints_bc_e4_nokpi_v2/bc_best.pt",
        "--no-kpi-features", "--no-kpi-aux",
        "--out", "checkpoints_dt_kpifix_no_kpi_v2"],
}


def finished(folder):
    path = EXP / folder / "loss_log.json"
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    return bool(rows) and rows[-1].get("epoch") == EPOCHS


def main():
    for name, extra in JOBS.items():
        out = extra[extra.index("--out") + 1]
        if finished(out):
            print(f"[{name}] 已完成，跳过", flush=True)
            continue
        print(f"\n[{name}] KPI实效化训练 -> {out}", flush=True)
        subprocess.run([
            sys.executable, "train_v2.py",
            "--epochs", str(EPOCHS), "--batch-size", "8",
            "--batches-per-epoch", "200", "--rtg-scale", "1000",
            "--explicit-kpi-branch", "--ablation", name, *extra,
        ], cwd=DT, check=True)
    print("完整/去KPI训练完成；运行 check_kpi_redesign_gate_v2.py", flush=True)


if __name__ == "__main__":
    main()
