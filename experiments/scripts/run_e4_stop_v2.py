"""为五个v2消融统一收集on-policy STOP监督并微调独立Stop Head。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
DT = ROOT / "dt_pruning"
VARIANTS = ("no_context", "no_gardening", "no_rq", "no_kpi", "no_rc")


def run(*args, cwd=ROOT):
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)


def stop_finished(root: Path):
    log = root / "log.json"
    best = root / "dt_stop_onpolicy_best.pt"
    if not log.exists() or not best.exists():
        return False
    with log.open(encoding="utf-8") as f:
        rows = json.load(f)
    return bool(rows) and rows[-1].get("epoch") == 100


def main():
    for name in VARIANTS:
        base = f"checkpoints_dt_e4_{name}_v2/dt_best.pt"
        if not (EXP / base).exists():
            raise SystemExit(f"缺少 {EXP / base}；先完成基础消融训练")
        data = f"data/stop_onpolicy_e4_{name}_v2"
        data_meta = EXP / data / "meta.json"
        if not data_meta.exists():
            print(f"\n[{name}] 收集自身rollout STOP监督", flush=True)
            run(
                "experiments/scripts/collect_onpolicy_stop_v2.py",
                "--checkpoint", base, "--candidate-cap", "8", "--out", data)
        else:
            print(f"[{name}] STOP监督已存在，跳过收集", flush=True)
        out = f"checkpoints_dt_e4_{name}_stop_v2"
        if stop_finished(EXP / out):
            print(f"[{name}] Stop Head已完成，跳过", flush=True)
            continue
        print(f"[{name}] 微调Stop Head", flush=True)
        run(
            "train_stop_onpolicy_v2.py", "--checkpoint", base,
            "--data", data, "--epochs", "100", "--steps-per-epoch", "50",
            "--batch-size", "32", "--out", out, cwd=DT)
    print("五个消融Stop Head完成；下一步运行 run_e4_eval_v2.py", flush=True)


if __name__ == "__main__":
    main()
