"""为既有v2轨迹补充候选级KPI真值，写入独立并列目录。

不改变动作序列，也不覆盖轨迹或图缓存。每个非STOP状态固定包含实际动作，
其余候选按固定种子抽样。KPI目标保存原始ΔK和剪枝比增量，训练时再统一缩放。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from config import CFG
from encoder import graph_arrays
from kpi_scales_v2 import activate_v2_kpi_scales
from reward import (KPI_KEYS, TreeContext, compute_branch_props, detect_crossing)
from trajectories_v2 import _candidate_subset, evaluate_candidates
from tree_data import BRANCH, TreeState
from v2_manifest import (CANDIDATE_KPI_ROOT, V2_TRAJ_ROOT,
                         assert_v2_write_path, load_manifest, manifest_hash)

SCHEMA_VERSION = 1
DEFAULT_CAP = 8


def _seed(tree: str, policy: str, variant: int, step: int) -> int:
    raw = f"candidate-kpi-v2:{tree}:{policy}:{variant}:{step}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def _trajectory_targets(tr: dict, cap: int, max_steps: int | None = None) -> dict:
    tree = tr["tree"]
    policy = tr["policy"]
    variant = int(tr.get("variant", 0))
    initial = tr["skeletons"][0]
    ctx = TreeContext(initial, CFG.reward, seed=tr["metadata"]["seed"])
    rows = []
    actions = tr["actions"] if max_steps is None else tr["actions"][:max_steps]
    for step, action in enumerate(actions):
        if action < 0:
            rows.append({
                "indices": np.zeros(0, dtype=np.int32),
                "dk": np.zeros((0, len(KPI_KEYS)), dtype=np.float32),
                "ratio_delta": np.zeros(0, dtype=np.float32),
            })
            continue
        sk = tr["skeletons"][step]
        state = TreeState(
            sk.copy(), int(tr["skeletons"][0].shape[0]),
            int(tr["metadata"]["initial_branch_count"]))
        props = compute_branch_props(sk, state.parent_map)
        crossing = detect_crossing(props, CFG.reward)
        k_before = {
            key: float(tr["kpis"][step][i])
            for i, key in enumerate(KPI_KEYS)
        }
        rng = np.random.default_rng(_seed(tree, policy, variant, step))
        legal = state.legal_actions(
            CFG.data.min_nodes_abs, CFG.data.min_nodes_ratio)
        subset = _candidate_subset(
            legal, rng, candidate_cap=cap, forced_bids=[int(action)])
        records = evaluate_candidates(
            state, props, k_before, crossing, ctx, rng,
            candidate_cap=len(subset), forced_bids=subset)
        arr = graph_arrays(
            sk, state.parent_map, state.pruning_ratio(), schema_version=2,
            tree_context=ctx, kpis=k_before)
        by_bid = {int(r["bid"]): r for r in records}
        ordered = [by_bid[b] for b in subset if b in by_bid]
        bids = np.asarray([r["bid"] for r in ordered], dtype=np.int64)
        indices = np.searchsorted(arr["bids"], bids).astype(np.int32)
        dk = np.asarray([
            [float(r["k_after"][k]) - k_before[k] for k in KPI_KEYS]
            for r in ordered
        ], dtype=np.float32)
        ratio_delta = np.asarray([
            float(r["ratio_after"]) - state.pruning_ratio() for r in ordered
        ], dtype=np.float32)
        if int(action) not in bids:
            raise ValueError(
                f"{tree}/{policy}/{variant} step={step} 实际动作未进入候选监督")
        chosen_pos = int(np.flatnonzero(bids == int(action))[0])
        expected_dk = (
            np.asarray(tr["kpis"][step + 1], dtype=np.float32)
            - np.asarray(tr["kpis"][step], dtype=np.float32))
        if not np.allclose(dk[chosen_pos], expected_dk, rtol=1e-5, atol=1e-6):
            raise ValueError(
                f"{tree}/{policy}/{variant} step={step} "
                "候选重放与原轨迹KPI不一致")
        rows.append({
            "indices": indices, "dk": dk, "ratio_delta": ratio_delta,
        })
    return {
        "policy": policy, "variant": variant, "steps": rows,
    }


def _worker(job: tuple[str, int]) -> tuple[str, list[dict] | None, str | None]:
    name, cap = job
    try:
        with (V2_TRAJ_ROOT / f"{name}.pkl").open("rb") as f:
            trajs = pickle.load(f)
        return name, [_trajectory_targets(tr, cap) for tr in trajs], None
    except Exception as exc:
        return name, None, repr(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "all"), default="all")
    ap.add_argument("--max-trees", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--candidate-cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 2, 1))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="只计算首棵首条轨迹的第一个状态，不写文件")
    args = ap.parse_args()
    if args.candidate_cap < 2:
        raise SystemExit("--candidate-cap 至少为2，必须包含实际动作和对照候选")

    activate_v2_kpi_scales()
    manifest = load_manifest()
    names = manifest["train"] if args.split == "train" else manifest["val"]
    if args.split == "all":
        names = manifest["train"] + manifest["val"]
    if args.only:
        names = [n for n in names if n == args.only]
    if args.max_trees is not None:
        names = names[:args.max_trees]
    if args.smoke:
        if not names:
            raise SystemExit("smoke没有可用树")
        with (V2_TRAJ_ROOT / f"{names[0]}.pkl").open("rb") as f:
            trajs = pickle.load(f)
        target = _trajectory_targets(trajs[0], args.candidate_cap, max_steps=1)
        step = target["steps"][0]
        print(
            f"smoke通过：{names[0]}/{target['policy']}，"
            f"候选数={len(step['indices'])}，dk={step['dk'].shape}",
            flush=True)
        return

    root = assert_v2_write_path(CANDIDATE_KPI_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "meta.json"
    expected_meta = {
        "schema_version": SCHEMA_VERSION,
        "manifest_hash": manifest_hash(manifest),
        "candidate_cap": args.candidate_cap,
        "kpi_keys": list(KPI_KEYS),
        "source_trajectories": str(V2_TRAJ_ROOT),
    }
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            old = json.load(f)
        if old != expected_meta:
            raise SystemExit(
                f"候选缓存配置不一致：{old} != {expected_meta}；"
                "请换目录或明确删除旧候选缓存")
    else:
        meta_path.write_text(
            json.dumps(expected_meta, ensure_ascii=False, indent=2),
            encoding="utf-8")

    todo = names if args.overwrite else [
        n for n in names if not (root / f"{n}.pkl").exists()]
    print(
        f"候选KPI并列缓存：计划{len(names)}棵，待生成{len(todo)}棵，"
        f"每状态最多{args.candidate_cap}候选 -> {root}",
        flush=True)
    started = time.time()
    with Pool(args.workers) as pool:
        jobs = [(name, args.candidate_cap) for name in todo]
        for i, (name, targets, err) in enumerate(
                pool.imap_unordered(_worker, jobs), 1):
            if err:
                print(f"[{i}/{len(todo)}] {name} 失败：{err}", flush=True)
                continue
            with (root / f"{name}.pkl").open("wb") as f:
                pickle.dump(targets, f, protocol=4)
            states = sum(
                bool(step["indices"].size)
                for tr in targets for step in tr["steps"])
            print(
                f"[{i}/{len(todo)}] {name} 完成，{states}个动作状态，"
                f"累计{(time.time() - started) / 60:.1f} min",
                flush=True)
    print(f"完成：{root}", flush=True)


if __name__ == "__main__":
    main()
