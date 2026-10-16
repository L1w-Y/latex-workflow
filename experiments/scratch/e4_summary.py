"""早期 E4 终端汇总。正式结果改用 experiments/scripts/e4_summary.py。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))
import e3_comprehensive as e3  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "dev"))
ABL = os.path.join(ROOT, "e4_ablation", "val")
VARIANTS = ["k1", "expert", "norq", "nori", "noprotect"]
END_KEYS = [
    "ratio", "kpi_sum", "protect_action_rate", "main_cuts", "constraint_rate",
    "autonomous_stop", "return", "points_per_cut",
]


def auc_of(item):
    pts = e3.curve_points(item)
    x = np.r_[0.0, e3.GRID]
    y = np.r_[0.0, [pts[float(q)]["kpi_sum"] for q in e3.GRID]]
    return float(np.trapezoid(y, x) / e3.GRID[-1]), {float(q): pts[float(q)]["kpi_sum"] for q in e3.GRID}


def show(label, a, b, rng, digits):
    st = e3.paired(a, b, rng)
    p = "NA" if st["p"] is None else f"{st['p']:.3g}"
    fmt = f"{{:.{digits}f}}"
    diff = fmt.format(st["diff"])
    lo, hi = fmt.format(st["ci"][0]), fmt.format(st["ci"][1])
    print(f"  {label}: mean {fmt.format(a.mean())}  diff {diff}  CI {lo},{hi}  p {p}  effect {st['effect']:+.2f}")


def main():
    rng = np.random.default_rng(0)
    ref_end = e3.read_result(os.path.join(ROOT, "stop_diag", "val"), "dt_base")
    ref_curve = e3.read_result(os.path.join(ROOT, "e3_comprehensive", "val"), "curve_dt")
    trees = sorted(set(ref_end) & set(ref_curve))
    ref_m = {t: e3.endpoint_metrics(ref_end[t], "dt") for t in trees}
    ref_auc = {}
    ref_grid = {q: [] for q in e3.GRID}
    for t in trees:
        a, g = auc_of(ref_curve[t])
        ref_auc[t] = a
        for q, v in g.items():
            ref_grid[q].append(v)
    print(f"FULL n={len(trees)} ratio={np.mean([ref_m[t]['ratio'] for t in trees]):.3f} "
          f"kpi={np.mean([ref_m[t]['kpi_sum'] for t in trees]):.1f} "
          f"auc={np.mean(list(ref_auc.values())):.1f} "
          f"protect={np.mean([ref_m[t]['protect_action_rate'] for t in trees]):.3f} "
          f"main={np.mean([ref_m[t]['main_cuts'] for t in trees]):.2f} "
          f"kpi30={np.mean(ref_grid[0.30]):.1f}")
    for name in VARIANTS:
        end = e3.read_result(ABL, name)
        curve = e3.read_result(ABL, "curve_" + name)
        ts = sorted(set(trees) & set(end) & set(curve))
        m = {t: e3.endpoint_metrics(end[t], "dt") for t in ts}
        auc, grid = {}, {q: [] for q in e3.GRID}
        for t in ts:
            a, g = auc_of(curve[t])
            auc[t] = a
            for q, v in g.items():
                grid[q].append(v)
        print(f"\n== {name} n={len(ts)}")
        for k in END_KEYS:
            digits = 3 if k in ("ratio", "protect_action_rate", "constraint_rate", "autonomous_stop") else 1
            show(k, np.array([m[t][k] for t in ts]), np.array([ref_m[t][k] for t in ts]), rng, digits)
        show("AUC", np.array([auc[t] for t in ts]), np.array([ref_auc[t] for t in ts]), rng, 1)
        for q in e3.GRID:
            show(f"KPI@{q:.2f}", np.array(grid[q]), np.array(ref_grid[q]), rng, 1)


if __name__ == "__main__":
    main()
