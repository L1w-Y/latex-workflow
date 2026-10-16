"""E3 基线对比汇总：剪后六项 KPI（主指标，全部策略）+ 累计回报（仅学习方法）+ 回报拆分。

读 results/<stage>/<exp>/<split>/ 下的 eval_<policy>.csv 与 actions_<policy>.json，输出同目录
e3_table.md 与 e3_table.json。

KPI 综合 = Σ_k W_k · ΔK_k / Δe_k（剪前→剪后整体变化，按奖励的权重与尺度加权，不逐刀截断），
即"整棵树 KPI 改善了多少"的单一数；β·KPI 综合与 β·ΣR_i 之差为逐刀截断损失。

用法（项目根目录）：
    python experiments/scripts/e3_table.py --exp e3_f5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))
from config import CFG  # noqa: E402

KEYS = ["hierarchy", "crowding", "balance", "light", "potential", "stability"]
CN = {"hierarchy": "层次", "crowding": "拥挤", "balance": "平衡", "light": "光照",
      "potential": "潜力", "stability": "稳定", "kpi_sum": "KPI综合"}
ORDER = ["dt", "expert", "bc", "bcq", "cql", "iql", "greedy", "random"]
LEARNED = ["dt", "bc", "bcq", "cql", "iql"]


def load(d: str, pol: str) -> dict[str, dict] | None:
    p = os.path.join(d, f"eval_{pol}.csv")
    if not os.path.exists(p):
        return None
    with open(os.path.join(d, f"actions_{pol}.json"), encoding="utf-8") as f:
        traces = json.load(f)
    cfg = CFG.reward
    out = {}
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tr = traces[row["tree"]]
            r = {k: float(row[f"{k}_delta"]) for k in KEYS}
            r["kpi_sum"] = sum(cfg.kpi_weights[k] * r[k] / cfg.kpi_scales[k] for k in KEYS)
            r["return"] = float(row["total_reward"])
            r["cuts"] = len(tr)
            r["ratio"] = float(row["final_pruning_ratio"])
            r["rq"] = cfg.alpha * sum(t["rq"] for t in tr)
            r["ri"] = cfg.beta * sum(t["ri"] for t in tr)
            r["rc"] = cfg.gamma * sum(t["rc"] for t in tr)
            r["clip_loss"] = cfg.beta * r["kpi_sum"] - r["ri"]
            r["pts_per_cut"] = np.mean([t["points_removed"] for t in tr]) if tr else 0.0
            r["main_cuts"] = sum("main_branch" in t["tags"] for t in tr)
            out[row["tree"]] = r
    return out


def paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    d = a - b
    nz = d[d != 0]
    p = float(wilcoxon(a, b).pvalue) if len(nz) >= 8 else None
    if len(nz):
        rk = rankdata(np.abs(nz))
        r = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum())   # 配对秩双列相关
    else:
        r = 0.0
    boot = rng.choice(d, size=(5000, len(d))).mean(1)
    return {"diff": float(d.mean()), "ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "win": float(np.mean(d > 0)), "p": p, "r": r}


def stars(p):
    return "" if p is None else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--stage", default="dev")
    ap.add_argument("--ref", default="dt")
    args = ap.parse_args()
    d = os.path.join(os.path.dirname(__file__), "..", "results", args.stage, args.exp, args.split)
    data = {p: t for p in ORDER if (t := load(d, p)) is not None}
    trees = sorted(set.intersection(*(set(t) for t in data.values())))
    arr = {p: {k: np.array([t[n][k] for n in trees]) for k in next(iter(t.values()))} for p, t in data.items()}
    rng = np.random.default_rng(0)
    cols = KEYS + ["kpi_sum"]
    L = [f"# E3 基线对比（{args.stage}/{args.exp}/{args.split}，{len(trees)} 棵）", ""]

    L += ["## 1. 剪后 KPI 变化（主指标，均值）", "",
          "| 策略 | " + " | ".join(CN[k] for k in cols) + " | 刀数 | 剪枝比 | 骨干枝刀/树（护栏） |",
          "|---" * (len(cols) + 4) + "|"]
    for p, a in arr.items():
        L.append(f"| {p} | " + " | ".join(f"{a[k].mean():+.4f}" if k != "kpi_sum" else f"{a[k].mean():+.1f}"
                                          for k in cols)
                 + f" | {a['cuts'].mean():.1f} | {a['ratio'].mean():.3f} | {a['main_cuts'].mean():.2f} |")

    ref = arr[args.ref]
    tests = {}
    L += ["", f"## 2. {args.ref} 对各策略：KPI 差值均值（显著性 * p<.05 ** p<.01 *** p<.001），胜率", "",
          "| 对手 | " + " | ".join(CN[k] for k in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for p, a in arr.items():
        if p == args.ref:
            continue
        tests[p] = {k: paired(ref[k], a[k], rng) for k in cols + ["return", "rq", "ri", "main_cuts"]}
        L.append(f"| {p} | " + " | ".join(
            f"{tests[p][k]['diff']:+.4f}{stars(tests[p][k]['p'])} ({tests[p][k]['win']:.0%})" if k != "kpi_sum"
            else f"{tests[p][k]['diff']:+.1f}{stars(tests[p][k]['p'])} ({tests[p][k]['win']:.0%})"
            for k in cols) + " |")

    L += ["", "## 3. 累计回报与拆分（回报只在学习方法间比较；专家等仅供参考）", "",
          "| 策略 | 回报 | 规则分 α·R_q | KPI 分 β·ΣR_i | 约束 γ·R_c | β·KPI综合 | 截断损失 | 每刀点数 |",
          "|---|---|---|---|---|---|---|---|"]
    for p, a in arr.items():
        mark = "" if p in LEARNED else "（参考）"
        L.append(f"| {p}{mark} | {a['return'].mean():.0f} | {a['rq'].mean():.0f} | {a['ri'].mean():.0f} | "
                 f"{a['rc'].mean():.1f} | {CFG.reward.beta * a['kpi_sum'].mean():.0f} | "
                 f"{a['clip_loss'].mean():.0f} | {a['pts_per_cut'].mean():.1f} |")
    L += ["", f"{args.ref} 对学习方法的回报配对检验：" + "；".join(
        f"{p} {tests[p]['return']['diff']:+.0f}{stars(tests[p]['return']['p'])}"
        for p in LEARNED if p in tests)]

    allr = [t for tt in data.values() for t in tt.values()]
    ppc = np.array([r["pts_per_cut"] for r in allr])
    eff = np.array([r["ri"] / (CFG.reward.beta * r["kpi_sum"]) if r["kpi_sum"] > 1e-6 else np.nan for r in allr])
    ok = np.isfinite(eff)
    L += ["", "## 4. 逐刀截断检查", "",
          f"KPI 分 / β·KPI 综合（>1 表示切碎多得，<1 表示大刀被截断）与每刀点数的相关："
          f"{np.corrcoef(ppc[ok], eff[ok])[0, 1]:.2f}；各策略中位："
          + "，".join(f"{p} {np.nanmedian([r['ri'] / (CFG.reward.beta * r['kpi_sum']) if r['kpi_sum'] > 1e-6 else np.nan for r in t.values()]):.2f}"
                     for p, t in data.items())]

    with open(os.path.join(d, "e3_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(d, "e3_table.json"), "w", encoding="utf-8") as f:
        json.dump({"trees": trees, "ref": args.ref, "tests": tests,
                   "means": {p: {k: float(v.mean()) for k, v in a.items()} for p, a in arr.items()}}, f, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
