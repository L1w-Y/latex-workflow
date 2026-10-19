"""统一评估KPI修复后的完整模型与五个核心消融。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
DT = ROOT / "dt_pruning"
VARIANTS = (
    "full", "no_context", "no_gardening", "no_rq", "no_kpi", "no_rc")
EXP_NAME = "e4_kpifix_v2"


def main():
    out_root = EXP / f"results/dev/{EXP_NAME}/val"
    for name in VARIANTS:
        checkpoint = (
            f"checkpoints_dt_e4_kpifix_{name}_stop_v2/"
            "dt_stop_onpolicy_best.pt")
        if not (EXP / checkpoint).exists():
            raise SystemExit(f"缺少 {EXP / checkpoint}")
        for curve in (False, True):
            tag = f"{name}_curve" if curve else name
            if (out_root / f"eval_{tag}.csv").exists():
                print(f"[{tag}] 已存在，跳过", flush=True)
                continue
            cmd = [
                sys.executable, "evaluate_v2.py", "--checkpoint", checkpoint,
                "--exp", EXP_NAME, "--tag", tag]
            if curve:
                cmd.append("--diagnose-stop-continuation")
            subprocess.run(cmd, cwd=DT, check=True)


if __name__ == "__main__":
    main()
