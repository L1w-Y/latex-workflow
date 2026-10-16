"""诊断奖励与六项 KPI 是否脱节。

两段，可分开跑：
  tree  已有 eval_*.csv：每项 KPI 剪前剪后、策略间配对检验（秒级）
  cuts  按 actions_*.json 重放剪枝，记录每刀六项 KPI 变化，按枝条标签汇总
        （验证树约 15–25 分钟，8 进程）

用法（dt_pruning 目录下）：
    python ../experiments/scripts/kpi_reward_diag.py --part tree
    python ../experiments/scripts/kpi_reward_diag.py --part cuts --workers 8
    python ../experiments/scripts/kpi_reward_diag.py --part both --workers 8

结果写入 results/dev/e3_baselines/val/kpi_diag_tree.json / kpi_diag_cuts.json
与 kpi_diag.md。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dt_pruning")))

from config import CFG  # noqa: E402
from evaluate import KPI_KEYS, _Env, result_dir, split_dirs  # noqa: E402
from tree_data import tree_name  # noqa: E402

HIGHER_BETTER = {k: True for k in KPI_KEYS}

TAG_ORDER = ["cross_inferior", "cross_other", "inward", "vigorous", "drooping", "weak",
             "protect_medium_short", "protect_sturdy", "main_branch", "none"]

PAIRS = [("dt", "expert"), ("dt", "random"), ("expert", "random"),
         ("iql", "dt"), ("iql", "expert"), ("cql", "dt"), ("bc", "expert")]


def _load_eval(d: str, policy: str) -> dict[str, dict]:
    path = os.path.join(d, f"eval_{policy}.csv")
    with open(path, encoding="utf-8") as f:
        return {r["tree"]: r for r in csv.DictReader(f)}


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    n = len(diff)
    if n < 8 or not np.any(diff != 0):
        p = None
    else:
        p = float(wilcoxon(a, b).pvalue)
    return {
        "n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "diff": float(diff.mean()), "win_rate": float((diff > 0).mean()),
        "p": p,
    }


def part_tree(d: str, policies: list[str]) -> dict:
    tables = {p: _load_eval(d, p) for p in policies if os.path.exists(os.path.join(d, f"eval_{p}.csv"))}
    trees = sorted(set.intersection(*[set(t) for t in tables.values()])) if tables else []
    out = {"n_trees": len(trees), "policies": list(tables), "before": {}, "delta": {},
           "after": {}, "delta_vs_zero": {}, "pairs": {}}

    first = next(iter(tables.values()))
    for k in KPI_KEYS:
        out["before"][k] = float(np.mean([float(first[t][f"{k}_before"]) for t in trees]))

    for p, table in tables.items():
        out["delta"][p] = {}
        out["after"][p] = {}
        out["delta_vs_zero"][p] = {}
        for k in KPI_KEYS:
            delta = np.array([float(table[t][f"{k}_delta"]) for t in trees])
            after = np.array([float(table[t][f"{k}_after"]) for t in trees])
            out["delta"][p][k] = float(delta.mean())
            out["after"][p][k] = float(after.mean())
            out["delta_vs_zero"][p][k] = _paired(delta, np.zeros_like(delta))
            out["delta_vs_zero"][p][k]["frac_improve"] = float((delta > 0).mean())

    for a, b in PAIRS:
        if a not in tables or b not in tables:
            continue
        out["pairs"][f"{a}_vs_{b}"] = {}
        for k in KPI_KEYS:
            va = np.array([float(tables[a][t][f"{k}_after"]) for t in trees])
            vb = np.array([float(tables[b][t][f"{k}_after"]) for t in trees])
            out["pairs"][f"{a}_vs_{b}"][k] = _paired(va, vb)
        ra = np.array([float(tables[a][t]["total_reward"]) for t in trees])
        rb = np.array([float(tables[b][t]["total_reward"]) for t in trees])
        out["pairs"][f"{a}_vs_{b}"]["total_reward"] = _paired(ra, rb)
    return out


def _fmt_p(p):
    if p is None:
        return "—"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.3f}"


def render_tree_md(rep: dict) -> str:
    lines = ["# KPI / 奖励诊断（树级）", "",
             f"共同树 {rep['n_trees']} 棵。六项均视为越高越好（拥挤度为枝间距均值）。", "",
             "## 剪前均值", ""]
    lines.append("| " + " | ".join(KPI_KEYS) + " |")
    lines.append("| " + " | ".join("---" for _ in KPI_KEYS) + " |")
    lines.append("| " + " | ".join(f"{rep['before'][k]:.3f}" for k in KPI_KEYS) + " |")
    lines += ["", "## 剪后 − 剪前（均值 / 改善比例 / 对 0 的 p）", "",
              "| 策略 | " + " | ".join(KPI_KEYS) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in KPI_KEYS) + " |")
    for p in rep["policies"]:
        cells = []
        for k in KPI_KEYS:
            z = rep["delta_vs_zero"][p][k]
            cells.append(f"{z['mean_a']:+.4f} ({z['frac_improve']:.0%} p={_fmt_p(z['p'])})")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    lines += ["", "## 剪后 KPI 配对（前 − 后；p 为 Wilcoxon）", ""]
    for pair, block in rep["pairs"].items():
        lines.append(f"### {pair}")
        lines.append("| 指标 | 均值差 | 胜率 | p |")
        lines.append("| --- | --- | --- | --- |")
        for k in KPI_KEYS + ["total_reward"]:
            r = block[k]
            lines.append(f"| {k} | {r['diff']:+.4f} | {r['win_rate']:.0%} | {_fmt_p(r['p'])} |")
        lines.append("")
    return "\n".join(lines)


# ---------- 逐刀重放 ----------

_CUTS_W = {}


def _init_cuts(data_root: str, seed: int):
    _CUTS_W.update(data_root=data_root, seed=seed)


def _replay_one(job: tuple[str, str, list[int]]) -> tuple[str, str, list[dict], str | None]:
    tree, policy, bids = job
    try:
        env = _Env(os.path.join(_CUTS_W["data_root"], tree), _CUTS_W["seed"])
        rows = []
        for bid in bids:
            k0 = dict(env.k)
            env.step(int(bid))
            cut = env.trace[-1]
            tags = cut["tags"] or ["none"]
            rows.append({
                "tree": tree, "policy": policy,
                "tags": tags, "rq": cut["rq"], "ri": cut["ri"], "rc": cut["rc"],
                "reward": cut["reward"],
                "points": cut["points_removed"],
                "dk": {k: float(env.k[k] - k0[k]) for k in KPI_KEYS},
            })
        return tree, policy, rows, None
    except Exception as e:
        return tree, policy, [], repr(e)


def _primary(tags: list[str]) -> str:
    for t in TAG_ORDER:
        if t in tags:
            return t
    return "none"


def part_cuts(d: str, policies: list[str], workers: int, limit: int | None) -> dict:
    trees = split_dirs("val")
    if limit:
        trees = trees[:limit]
    names = [tree_name(t) for t in trees]
    jobs = []
    for p in policies:
        with open(os.path.join(d, f"actions_{p}.json"), encoding="utf-8") as f:
            acts = json.load(f)
        for n in names:
            bids = [c["bid"] for c in acts[n]]
            jobs.append((n, p, bids))
    print(f"逐刀重放 {len(jobs)} 份（{len(names)} 棵 × {len(policies)} 策略），"
          f"{workers} 进程", flush=True)
    t0 = time.time()
    rows = []
    failed = []
    initargs = (CFG.data.data_root, CFG.eval.seed)
    if workers <= 1:
        _init_cuts(*initargs)
        it = map(_replay_one, jobs)
        pool = None
    else:
        pool = Pool(workers, initializer=_init_cuts, initargs=initargs)
        it = pool.imap_unordered(_replay_one, jobs)
    for i, (tree, policy, recs, err) in enumerate(it, 1):
        if err:
            failed.append(f"{policy}/{tree}: {err}")
            print(f"[{i}/{len(jobs)}] {policy} {tree} 失败: {err}", flush=True)
        else:
            rows.extend(recs)
            print(f"[{i}/{len(jobs)}] {policy} {tree} {len(recs)} 刀 "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if pool is not None:
        pool.close(); pool.join()

    by_tag = {p: {} for p in policies}
    by_pol = {}
    for p in policies:
        pr = [r for r in rows if r["policy"] == p]
        by_pol[p] = _summarize_cuts(pr)
        grouped = defaultdict(list)
        for r in pr:
            grouped[_primary(r["tags"])].append(r)
        by_tag[p] = {tag: _summarize_cuts(v) for tag, v in grouped.items()}
    return {"n_trees": len(names), "n_cuts": len(rows), "failed": failed,
            "minutes": (time.time() - t0) / 60, "by_policy": by_pol, "by_tag": by_tag}


def _summarize_cuts(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    dk = {k: float(np.mean([r["dk"][k] for r in recs])) for k in KPI_KEYS}
    frac = {k: float(np.mean([r["dk"][k] > 0 for r in recs])) for k in KPI_KEYS}
    return {
        "n": len(recs),
        "reward": float(np.mean([r["reward"] for r in recs])),
        "rq": float(np.mean([r["rq"] for r in recs])),
        "ri": float(np.mean([r["ri"] for r in recs])),
        "points": float(np.mean([r["points"] for r in recs])),
        "dk": dk, "frac_up": frac,
    }


def render_cuts_md(rep: dict) -> str:
    lines = ["# KPI / 奖励诊断（逐刀）", "",
             f"{rep['n_trees']} 棵，{rep['n_cuts']} 刀，用时 {rep['minutes']:.1f} min。",
             "主标签按判别顺序取第一个；dk 为该刀引起的 KPI 变化。", ""]
    lines += ["## 按策略", "",
              "| 策略 | 刀数 | 单刀 R | R_q | R_i | 点数 | " + " | ".join(KPI_KEYS) + " |"]
    lines.append("| --- | --- | --- | --- | --- | --- | " + " | ".join("---" for _ in KPI_KEYS) + " |")
    for p, s in rep["by_policy"].items():
        if not s.get("n"):
            continue
        dk = " | ".join(f"{s['dk'][k]:+.4f}" for k in KPI_KEYS)
        lines.append(f"| {p} | {s['n']} | {s['reward']:.1f} | {s['rq']:.2f} | {s['ri']:.2f} | "
                     f"{s['points']:.1f} | {dk} |")
    lines += ["", "## 按主标签（专家 / DT / 随机）", ""]
    for p in ("expert", "dt", "random", "iql"):
        if p not in rep["by_tag"]:
            continue
        lines.append(f"### {p}")
        lines.append("| 标签 | 刀数 | R_q | R_i | " + " | ".join(KPI_KEYS) + " |")
        lines.append("| --- | --- | --- | --- | " + " | ".join("---" for _ in KPI_KEYS) + " |")
        for tag in TAG_ORDER:
            s = rep["by_tag"][p].get(tag)
            if not s or not s.get("n"):
                continue
            dk = " | ".join(f"{s['dk'][k]:+.4f}" for k in KPI_KEYS)
            lines.append(f"| {tag} | {s['n']} | {s['rq']:.2f} | {s['ri']:.2f} | {dk} |")
        lines.append("")
    if rep["failed"]:
        lines += ["## 失败", ""] + [f"- {x}" for x in rep["failed"]]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e3_baselines")
    ap.add_argument("--stage", default="dev")
    ap.add_argument("--split", default="val")
    ap.add_argument("--part", choices=["tree", "cuts", "both"], default="both")
    ap.add_argument("--policies", default="expert,dt,iql,cql,bcq,bc,random,greedy")
    ap.add_argument("--cut-policies", default="expert,dt,random,iql",
                    help="逐刀重放的策略；默认专家/DT/随机/IQL")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 棵（冒烟）")
    args = ap.parse_args()

    d = result_dir(args.stage, args.exp, args.split)
    policies = [p for p in args.policies.split(",") if p]
    md_parts = []

    if args.part in ("tree", "both"):
        tree = part_tree(d, policies)
        with open(os.path.join(d, "kpi_diag_tree.json"), "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        md_parts.append(render_tree_md(tree))
        print("已写 kpi_diag_tree.json")

    if args.part in ("cuts", "both"):
        cuts = part_cuts(d, [p for p in args.cut_policies.split(",") if p],
                         args.workers, args.limit)
        with open(os.path.join(d, "kpi_diag_cuts.json"), "w", encoding="utf-8") as f:
            json.dump(cuts, f, indent=2, ensure_ascii=False)
        md_parts.append(render_cuts_md(cuts))
        print("已写 kpi_diag_cuts.json")

    path = os.path.join(d, "kpi_diag.md")
    existing = ""
    if args.part != "both" and os.path.exists(path):
        existing = open(path, encoding="utf-8").read()
        if args.part == "tree" and "# KPI / 奖励诊断（逐刀）" in existing:
            existing = existing.split("# KPI / 奖励诊断（逐刀）")[1]
            md_parts.append("# KPI / 奖励诊断（逐刀）" + existing)
        elif args.part == "cuts" and "# KPI / 奖励诊断（树级）" in existing:
            head = existing.split("# KPI / 奖励诊断（逐刀）")[0]
            md_parts = [head.rstrip(), *md_parts]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_parts).strip() + "\n")
    print("已写", path)


if __name__ == "__main__":
    main()
