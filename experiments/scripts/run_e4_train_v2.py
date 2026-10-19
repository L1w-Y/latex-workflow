"""训练 v2 E4 的五个核心消融（不执行耗时随机轨迹生成）。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DT = ROOT / "dt_pruning"
EXP = ROOT / "experiments"
EPOCHS = 60
BC = "checkpoints_bc_v2/bc_best.pt"
BC_NOKPI = "checkpoints_bc_e4_nokpi_v2/bc_best.pt"

JOBS = {
    "no_context": [
        "--context-len", "1", "--reward-set", "full",
        "--policies", "gardening,topk,kpi", "--init-from", BC],
    "no_gardening": [
        "--context-len", "16", "--reward-set", "norq",
        "--policies", "random", "--knowledge-weight", "0",
        "--cache-root", "data/graph_cache_v2_e4_noknowledge",
        "--no-rule-features"],
    "no_rq": [
        "--context-len", "16", "--reward-set", "norq",
        "--policies", "gardening,topk,kpi", "--init-from", BC],
    "no_kpi": [
        "--context-len", "16", "--reward-set", "nori",
        "--policies", "gardening,topk", "--init-from", BC_NOKPI,
        "--no-kpi-features", "--no-kpi-aux"],
    "no_rc": [
        "--context-len", "16", "--reward-set", "norc",
        "--policies", "gardening,topk,kpi", "--init-from", BC],
}


def run(*args):
    subprocess.run([sys.executable, *args], cwd=DT, check=True)


def finished(folder: Path, epochs: int, log_name="loss_log.json"):
    log = folder / log_name
    if not log.exists():
        return False
    with log.open(encoding="utf-8") as f:
        rows = json.load(f)
    return bool(rows) and rows[-1].get("epoch") == epochs


def preflight():
    required = [
        EXP / BC,
        EXP / "data/graph_cache_v2_e4_noknowledge/rtg_norq/index.json",
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise SystemExit(
            "缺少前置文件：\n" + "\n".join(missing)
            + "\n先运行 prepare_e4_noknowledge_v2.py")


def main():
    preflight()
    for reward_set in ("norq", "nori", "norc"):
        index = EXP / f"data/graph_cache_v2_curated/rtg_{reward_set}/index.json"
        if not index.exists():
            print(f"生成 rtg_{reward_set}（只重组缓存，不重算KPI）", flush=True)
            run("reward_parts_v2.py", "apply", "--set", reward_set)

    bc_out = EXP / "checkpoints_bc_e4_nokpi_v2"
    if not finished(bc_out, 30, "bc_log.json"):
        print("训练去KPI专用BC初始化器", flush=True)
        run(
            "train_bc_v2.py", "--epochs", "30", "--batch-size", "64",
            "--policies", "gardening,topk", "--no-kpi-features",
            "--out", "checkpoints_bc_e4_nokpi_v2")

    for name, variant_args in JOBS.items():
        out = f"checkpoints_dt_e4_{name}_v2"
        if finished(EXP / out, EPOCHS):
            print(f"[{name}] 已完成，跳过", flush=True)
            continue
        print(f"\n[{name}] 开始训练", flush=True)
        run(
            "train_v2.py", "--epochs", str(EPOCHS), "--batch-size", "8",
            "--batches-per-epoch", "200", "--rtg-scale", "1000",
            "--ablation", name, "--out", out, *variant_args)
    print("五个基础消融训练完成；下一步运行 run_e4_stop_v2.py", flush=True)


if __name__ == "__main__":
    main()
