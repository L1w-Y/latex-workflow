"""在31棵验证树上统一评估v2完整DT与五个核心消融。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
DT = ROOT / "dt_pruning"
VARIANTS = {
    "full": "checkpoints_dt_stop_onpolicy_v2/dt_stop_onpolicy_best.pt",
    "no_context": "checkpoints_dt_e4_no_context_stop_v2/dt_stop_onpolicy_best.pt",
    "no_gardening": "checkpoints_dt_e4_no_gardening_stop_v2/dt_stop_onpolicy_best.pt",
    "no_rq": "checkpoints_dt_e4_no_rq_stop_v2/dt_stop_onpolicy_best.pt",
    "no_kpi": "checkpoints_dt_e4_no_kpi_stop_v2/dt_stop_onpolicy_best.pt",
    "no_rc": "checkpoints_dt_e4_no_rc_stop_v2/dt_stop_onpolicy_best.pt",
}
EXP_NAME = "e4_redesign_v2"


def main():
    out_root = EXP / f"results/dev/{EXP_NAME}/val"
    for name, checkpoint in VARIANTS.items():
        if not (EXP / checkpoint).exists():
            raise SystemExit(f"缺少 {EXP / checkpoint}")
        for curve in (False, True):
            tag = f"{name}_curve" if curve else name
            output = out_root / f"eval_{tag}.csv"
            if output.exists():
                print(f"[{tag}] 已存在，跳过", flush=True)
                continue
            cmd = [
                sys.executable, "evaluate_v2.py", "--checkpoint", checkpoint,
                "--exp", EXP_NAME, "--tag", tag,
            ]
            if curve:
                cmd.append("--diagnose-stop-continuation")
            print(f"\n[{tag}] 开始31树评估", flush=True)
            subprocess.run(cmd, cwd=DT, check=True)
    print(f"评估完成：{out_root}", flush=True)


if __name__ == "__main__":
    main()
