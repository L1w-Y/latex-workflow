"""候选级KPI修复的最小full/no_kpi重训；不扩展其他消融。"""
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
        "--monotonic-kpi-branch",
        "--candidate-kpi-root", "data/candidate_kpi_v2_curated",
        "--out", "checkpoints_dt_kpicandidate_v2"],
    "no_kpi": [
        "--context-len", "16", "--reward-set", "nori",
        "--policies", "gardening,topk",
        "--init-from", "checkpoints_bc_e4_nokpi_v2/bc_best.pt",
        "--no-kpi-features", "--no-kpi-aux", "--monotonic-kpi-branch",
        "--out", "checkpoints_dt_kpicandidate_no_kpi_v2"],
}


def finished(folder: str) -> bool:
    path = EXP / folder / "loss_log.json"
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    return bool(rows) and rows[-1].get("epoch") == EPOCHS


def main() -> None:
    candidate_meta = EXP / "data/candidate_kpi_v2_curated/meta.json"
    if not candidate_meta.exists():
        raise SystemExit(
            "缺候选KPI缓存；先运行 "
            "python -u dt_pruning/candidate_kpi_v2.py --workers <N>")
    for name, extra in JOBS.items():
        out = extra[extra.index("--out") + 1]
        if finished(out):
            print(f"[{name}] 已完成，跳过", flush=True)
            continue
        print(f"\n[{name}] 候选级KPI训练 -> {out}", flush=True)
        subprocess.run([
            sys.executable, "train_v2.py",
            "--epochs", str(EPOCHS), "--batch-size", "8",
            "--batches-per-epoch", "200", "--rtg-scale", "1000",
            "--ablation", name, *extra,
        ], cwd=DT, check=True)
    print("最小full/no_kpi训练完成；下一步运行候选级预设门槛", flush=True)


if __name__ == "__main__":
    main()
