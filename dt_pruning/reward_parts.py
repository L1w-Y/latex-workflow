"""轨迹奖励分项：重算（compute）→ 标定 R_i 尺度（calibrate）→ 写入回报旁路文件（apply）。

专家动作只取决于枝条判别规则（不调用 KPI），随机动作与奖励无关，因此修改 KPI 口径或奖励权重时
轨迹的状态与动作不变，只需按分项重组每步奖励，不重新生成轨迹。

compute：逐步记录（未加权）
    rq_bad      枝条质量奖励中的不良枝部分（R_q > 0 时即 R_q）
    rq_protect  功能枝保护惩罚（R_q < 0 时即 R_q）
    rc          约束惩罚
    dk[6]       六项 KPI 的单步变化（顺序同 reward.KPI_KEYS）
    pts         每刀剪除骨架点数（R_q 按剪除量缩放用）
    stop        终止奖励（STOP 步）
  输出 graph_cache/reward_parts/<树名>.pkl：[{"policy", "pts", "parts": float32[T-1, 3], "dk": float32[T-1, 6], "stop"}]
  顺序同轨迹文件 / 图缓存。

addpts：给 compute 已生成、缺 pts 的分项文件补剪除点数（只读轨迹，不重算 KPI）。

calibrate：Δe_k = 训练树轨迹（专家 + 随机）单刀 |ΔK_k| 的 95 分位；rq_volume_ref = 训练树专家每刀平均剪除点数。
  写 graph_cache/kpi_scales.json，并打印可贴入 config.py 的数值。

apply：按 config 当前权重与尺度重组奖励，写 graph_cache/rtg_<set>/<树名>.pkl（每条轨迹的 rtg 与总回报）
  与 rtg_<set>/index.json；graph_cache.load_cache 读取时用它覆盖缓存中的旧回报。

用法（dt_pruning 目录下）：
    python reward_parts.py compute --workers 12
    python reward_parts.py addpts --workers 6
    python reward_parts.py calibrate
    python reward_parts.py apply --set full
    python reward_parts.py compute --max-trees 2 --workers 2      # 冒烟测试
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from config import CFG
from graph_cache import CACHE_ROOT, TRAJ_ROOT, available_train_names, load_val_split
from reward import (KPI_KEYS, TreeContext, branch_quality_reward, branch_context,
                    compute_branch_props, compute_kpis, constraint_penalty, detect_crossing)
from tree_data import build_parent_map

PARTS_ROOT = os.path.join(CACHE_ROOT, "reward_parts")
SCALES_PATH = os.path.join(CACHE_ROOT, "kpi_scales.json")
PART_KEYS = ("rq_bad", "rq_protect", "rc")


def rtg_root(name: str) -> str:
    return os.path.join(CACHE_ROOT, f"rtg_{name}")


def step_rewards(rec: dict, cfg=None, alpha=None, beta=None, gamma=None,
                 use_rq=True, use_protect=True, use_ri=True, use_rc=True) -> np.ndarray:
    """分项 → 每步奖励 [T]（最后一步为终止奖励）；默认取 config 当前权重。"""
    cfg = cfg or CFG.reward
    a = cfg.alpha if alpha is None else alpha
    b = cfg.beta if beta is None else beta
    g = cfg.gamma if gamma is None else gamma
    p, dk = rec["parts"], rec["dk"]
    rq = (p[:, 0] if use_rq else 0.0) + (p[:, 1] if use_protect else 0.0)
    if cfg.rq_by_volume:
        rq = rq * rec["pts"] / cfg.rq_volume_ref
    w = np.array([cfg.kpi_weights[k] for k in KPI_KEYS])
    s = np.array([cfg.kpi_scales[k] for k in KPI_KEYS])
    ri = (dk / s * w).sum(axis=1) if use_ri else 0.0
    rc = p[:, 2] if use_rc else 0.0
    r = a * rq + b * ri + g * rc
    return np.append(r, rec["stop"]).astype(np.float32)


# ------------------------------------------------------------------ compute

def traj_parts(tr: dict) -> dict:
    cfg = CFG.reward
    sks, pms, acts = tr["skeletons"], tr.get("parent_maps"), tr["actions"]
    ctx = TreeContext(sks[0], cfg, seed=tr["metadata"]["seed"])
    pm = (lambda t: pms[t]) if pms else (lambda t: build_parent_map(sks[t]))
    props = compute_branch_props(sks[0], pm(0))
    k = compute_kpis(sks[0], props, cfg, ctx)
    parts, dks = [], []
    for t, a in enumerate(acts[:-1]):
        crossing = detect_crossing(props, cfg)
        rq = branch_quality_reward(a, props, crossing, *branch_context(sks[t], props, cfg), cfg)
        rc = constraint_penalty(a, props, cfg)
        props_after = compute_branch_props(sks[t + 1], pm(t + 1))
        k_after = compute_kpis(sks[t + 1], props_after, cfg, ctx)
        parts.append((max(rq, 0.0), min(rq, 0.0), rc))
        dks.append([k_after[key] - k[key] for key in KPI_KEYS])
        props, k = props_after, k_after
    return {"policy": tr["policy"],
            "pts": traj_pts(tr),
            "parts": np.asarray(parts, dtype=np.float32).reshape(-1, 3),
            "dk": np.asarray(dks, dtype=np.float32).reshape(-1, len(KPI_KEYS)),
            "stop": float(tr["rewards"][-1])}


def traj_pts(tr: dict) -> np.ndarray:
    """每刀剪除的骨架点数 [T-1]。"""
    n = np.array([len(s) for s in tr["skeletons"]])
    return (n[:-1] - n[1:]).astype(np.float32)


def _pts_worker(name: str):
    try:
        recs = load_parts(name)
        if recs is None:
            return name, "缺分项文件"
        with open(os.path.join(TRAJ_ROOT, f"{name}.pkl"), "rb") as f:
            trajs = pickle.load(f)
        for r, tr in zip(recs, trajs):
            pts = traj_pts(tr)
            if r["policy"] != tr["policy"] or len(pts) != len(r["parts"]):
                return name, "与轨迹不对齐"
            r["pts"] = pts
        with open(os.path.join(PARTS_ROOT, f"{name}.pkl"), "wb") as f:
            pickle.dump(recs, f, protocol=4)
        return name, None
    except Exception as e:
        return name, repr(e)


def cmd_addpts(args):
    todo = [n for n in available_train_names() if (r := load_parts(n)) is not None and "pts" not in r[0]]
    print(f"待补剪除点数 {len(todo)} 棵", flush=True)
    t0, failed = time.time(), []
    with Pool(args.workers) as pool:
        for k, (name, err) in enumerate(pool.imap_unordered(_pts_worker, todo), 1):
            if err:
                failed.append(name)
                print(f"[{k}/{len(todo)}] {name} 失败: {err}", flush=True)
            elif k % 20 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}] 已耗时 {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"完成；失败 {len(failed)} 棵{'：' + ','.join(failed) if failed else ''}")


def _worker(name: str):
    try:
        with open(os.path.join(TRAJ_ROOT, f"{name}.pkl"), "rb") as f:
            trajs = pickle.load(f)
        out = [traj_parts(tr) for tr in trajs]
        with open(os.path.join(PARTS_ROOT, f"{name}.pkl"), "wb") as f:
            pickle.dump(out, f, protocol=4)
        return name, None
    except Exception as e:
        return name, repr(e)


def load_parts(name: str) -> list[dict] | None:
    p = os.path.join(PARTS_ROOT, f"{name}.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        recs = pickle.load(f)
    return recs if recs and "dk" in recs[0] else None


def cmd_compute(args):
    os.makedirs(PARTS_ROOT, exist_ok=True)
    names = available_train_names()
    todo = names[:args.max_trees] if args.max_trees else names
    if not args.overwrite:
        todo = [n for n in todo if load_parts(n) is None]
    print(f"待处理 {len(todo)} 棵（训练 + 验证树），输出 {PARTS_ROOT}", flush=True)
    t0, failed = time.time(), []
    with Pool(args.workers) as pool:
        for k, (name, err) in enumerate(pool.imap_unordered(_worker, todo), 1):
            if err:
                failed.append(name)
                print(f"[{k}/{len(todo)}] {name} 失败: {err}", flush=True)
            else:
                print(f"[{k}/{len(todo)}] {name} 完成，已耗时 {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"完成；失败 {len(failed)} 棵{'：' + ','.join(failed) if failed else ''}")


# ---------------------------------------------------------------- calibrate

def cmd_calibrate(args):
    train = load_val_split()["train"]
    parts = {n: load_parts(n) for n in train}
    missing = [n for n, p in parts.items() if p is None]
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)}/{len(train)} 棵训练树缺新分项文件：请先运行 compute")
    dk = [r["dk"] for p in parts.values() if p for r in p]
    if not dk:
        raise SystemExit("缺分项文件：请先运行 compute")
    dk = np.abs(np.concatenate(dk))
    q = {k: float(np.quantile(dk[:, i], args.quantile)) for i, k in enumerate(KPI_KEYS)}
    med = {k: float(np.median(dk[:, i])) for i, k in enumerate(KPI_KEYS)}
    zero = {k: float(np.mean(dk[:, i] == 0)) for i, k in enumerate(KPI_KEYS)}
    pts = [r["pts"] for p in parts.values() if p for r in p if r["policy"] == "expert" and "pts" in r]
    ref = float(np.concatenate(pts).mean()) if pts else None
    out = {"quantile": args.quantile, "n_steps": int(len(dk)), "scales": q, "median_abs": med,
           "frac_zero": zero, "train_trees": len(train), "rq_volume_ref": ref}
    with open(SCALES_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"训练树 {len(train)} 棵，{len(dk)} 刀；|ΔK| 的 {args.quantile:.0%} 分位：")
    for k in KPI_KEYS:
        print(f"  {k:<10} Δe = {q[k]:.3e}   中位 {med[k]:.2e}   为 0 的刀 {zero[k]:.1%}")
    print("贴入 config.py RewardConfig.kpi_scales：")
    print("    " + ", ".join(f'"{k}": {q[k]:.2e}' for k in KPI_KEYS))
    if ref is None:
        print("缺剪除点数：先运行 addpts，否则无法标定 rq_volume_ref")
    else:
        print(f"贴入 config.py RewardConfig.rq_volume_ref：{ref:.1f}（训练树专家每刀平均剪除点数）")
    print("写入", SCALES_PATH)


# -------------------------------------------------------------------- apply

def cmd_apply(args):
    expected = {
        "full": (False, False, False, False),
        "norq": (True, True, False, False),
        "noprotect": (False, True, False, False),
        "nori": (False, False, True, False),
        "norc": (False, False, False, True),
    }
    flags = (args.no_rq, args.no_protect, args.no_ri, args.no_rc)
    if args.set in expected and flags != expected[args.set]:
        raise SystemExit(
            f"回报集 {args.set} 的开关应为 no_rq, no_protect, no_ri, no_rc = {expected[args.set]}，"
            f"实际为 {flags}。去掉整个规则项需要同时加 --no-rq --no-protect"
        )
    cfg = CFG.reward
    if os.path.exists(SCALES_PATH) and not args.allow_partial:
        with open(SCALES_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        cal, ref = saved["scales"], saved.get("rq_volume_ref")
        off = [k for k in KPI_KEYS if not np.isclose(cal[k], cfg.kpi_scales[k], rtol=0.02)]
        if cfg.rq_by_volume and (ref is None or not np.isclose(ref, cfg.rq_volume_ref, rtol=0.02)):
            off.append("rq_volume_ref")
        if off:
            raise SystemExit(f"config 与 {SCALES_PATH} 不一致（{off}）：先运行 addpts / calibrate 并把标定值贴入 config")
    root = rtg_root(args.set)
    os.makedirs(root, exist_ok=True)
    index, missing = {}, []
    for n in available_train_names():
        recs = load_parts(n)
        if recs is None:
            missing.append(n)
            continue
        out = []
        for r in recs:
            rew = step_rewards(r, use_rq=not args.no_rq, use_protect=not args.no_protect,
                               use_ri=not args.no_ri, use_rc=not args.no_rc)
            out.append({"policy": r["policy"], "rtg": np.cumsum(rew[::-1])[::-1].astype(np.float32),
                        "total_reward": float(rew.sum())})
        with open(os.path.join(root, f"{n}.pkl"), "wb") as f:
            pickle.dump(out, f, protocol=4)
        index[n] = [{"policy": o["policy"], "total_reward": o["total_reward"], "steps": len(o["rtg"])}
                    for o in out]
    meta = {"set": args.set, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_rq": not args.no_rq, "use_protect": not args.no_protect,
            "use_ri": not args.no_ri, "use_rc": not args.no_rc,
            "alpha": cfg.alpha, "beta": cfg.beta, "gamma": cfg.gamma,
            "kpi_weights": cfg.kpi_weights, "kpi_scales": cfg.kpi_scales,
            "rq_by_volume": cfg.rq_by_volume, "rq_volume_ref": cfg.rq_volume_ref}
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "trees": index}, f)
    for stale in (
        os.path.join(CACHE_ROOT, f"expert_return_stats_{args.set}.json"),
        os.path.join(CACHE_ROOT, f"return_stats_{args.set}_random.json"),
    ):
        if os.path.exists(stale):
            os.remove(stale)
    print(f"写入 {root}：{len(index)} 棵；缺分项 {len(missing)} 棵")
    tot = [t["total_reward"] for v in index.values() for t in v if t["policy"] == "expert"]
    if tot:
        print(f"专家轨迹总回报均值 {np.mean(tot):.1f}（{len(tot)} 条）")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--workers", type=int, default=max(os.cpu_count() - 4, 1))
    c.add_argument("--max-trees", type=int, default=None)
    c.add_argument("--overwrite", action="store_true")
    c1 = sub.add_parser("addpts", help="给已有分项文件补每刀剪除点数（只读轨迹，不重算 KPI）")
    c1.add_argument("--workers", type=int, default=6)
    c2 = sub.add_parser("calibrate")
    c2.add_argument("--quantile", type=float, default=0.95)
    c2.add_argument("--allow-partial", action="store_true", help="冒烟测试：允许部分树缺分项")
    c3 = sub.add_parser("apply")
    c3.add_argument("--set", default="full")
    c3.add_argument("--no-rq", action="store_true", help="去掉不良枝得分（正的规则项）")
    c3.add_argument("--no-protect", action="store_true", help="去掉功能枝保护惩罚")
    c3.add_argument("--no-ri", action="store_true", help="去掉指标项 R_i")
    c3.add_argument("--no-rc", action="store_true", help="去掉约束项 R_c")
    c3.add_argument("--allow-partial", action="store_true", help="冒烟测试：跳过尺度一致性检查")
    args = ap.parse_args()
    {"compute": cmd_compute, "addpts": cmd_addpts, "calibrate": cmd_calibrate,
     "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
