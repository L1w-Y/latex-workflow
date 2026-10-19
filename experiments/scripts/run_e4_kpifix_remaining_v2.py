"""KPI门槛通过后训练其余四个E4消融。"""
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
    "no_context": [
        "--context-len", "1", "--reward-set", "full",
        "--policies", "gardening,topk,kpi",
        "--init-from", "checkpoints_bc_v2/bc_best.pt"],
    "no_gardening": [
        "--context-len", "16", "--reward-set", "norq",
        "--policies", "random", "--knowledge-weight", "0",
        "--cache-root", "data/graph_cache_v2_e4_noknowledge",
        "--no-rule-features"],
    "no_rq": [
        "--context-len", "16", "--reward-set", "norq",
        "--policies", "gardening,topk,kpi",
        "--init-from", "checkpoints_bc_v2/bc_best.pt"],
    "no_rc": [
        "--context-len", "16", "--reward-set", "norc",
        "--policies", "gardening,topk,kpi",
        "--init-from", "checkpoints_bc_v2/bc_best.pt"],
}


def finished(out):
    path = EXP / out / "loss_log.json"
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    return bool(rows) and rows[-1].get("epoch") == EPOCHS


def main():
    gate_path = EXP / "results/dev/kpi_redesign_v2/gate.json"
    if not gate_path.exists() or not json.loads(
            gate_path.read_text(encoding="utf-8")).get("passed"):
        raise SystemExit("KPI门槛未通过，禁止扩展训练其余消融")
    print("重算去园艺知识RTG：只读现有缓存，不重新生成轨迹", flush=True)
    subprocess.run([
        sys.executable, "reward_parts_v2.py", "apply", "--set", "norq",
        "--cache-root", "data/graph_cache_v2_e4_noknowledge", "--overwrite",
    ], cwd=DT, check=True)
    for name, extra in JOBS.items():
        out = f"checkpoints_dt_e4_kpifix_{name}_v2"
        if finished(out):
            print(f"[{name}] 已完成，跳过", flush=True)
            continue
        subprocess.run([
            sys.executable, "train_v2.py",
            "--epochs", str(EPOCHS), "--batch-size", "8",
            "--batches-per-epoch", "200", "--rtg-scale", "1000",
            "--explicit-kpi-branch", "--ablation", name,
            "--out", out, *extra,
        ], cwd=DT, check=True)


if __name__ == "__main__":
    main()
