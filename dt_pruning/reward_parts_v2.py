"""schema=2 KPI 标定与回报旁路。

图缓存已含原始 ΔK、R_q、连续 R_c 和剪除成本，因此无需再次计算昂贵 KPI。
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np

from config import CFG
from reward import KPI_KEYS
from v2_manifest import V2_CACHE_ROOT, assert_v2_write_path, load_manifest, manifest_hash


SCALES_PATH = V2_CACHE_ROOT / "kpi_scales.json"
SETS = {
    "full": (True, True, True, True),
    "norq": (False, False, True, True),
    "noprotect": (True, False, True, True),
    "nori": (True, True, False, True),
    "norc": (True, True, True, False),
}


def _load_tree(name: str, cache_root=V2_CACHE_ROOT) -> list[dict]:
    path = cache_root / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        return pickle.load(f)


def calibrate(quantile: float, allow_partial: bool) -> None:
    manifest = load_manifest()
    arrays, missing = [], []
    for name in manifest["train"]:
        try:
            for tr in _load_tree(name, V2_CACHE_ROOT):
                arrays.append(np.abs(tr["dk"][tr["dk_mask"]]))
        except FileNotFoundError:
            missing.append(name)
    if missing and not allow_partial:
        raise SystemExit(f"{len(missing)} 棵训练树缺 v2 cache；先完成 graph_cache_v2.py")
    arrays = [x for x in arrays if len(x)]
    if not arrays:
        raise SystemExit("没有可标定的 ΔK")
    values = np.concatenate(arrays)
    # 防止某一维全零导致除零；该情况仍在报告中明确记录。
    scales = {
        key: max(float(np.quantile(values[:, i], quantile)), 1e-8)
        for i, key in enumerate(KPI_KEYS)
    }
    output = {
        "schema_version": 2, "manifest_hash": manifest_hash(manifest),
        "quantile": quantile, "n_steps": int(len(values)),
        "train_trees_used": len(manifest["train"]) - len(missing),
        "missing": missing, "scales": scales,
        "median_abs": {
            key: float(np.median(values[:, i])) for i, key in enumerate(KPI_KEYS)},
        "frac_zero": {
            key: float(np.mean(values[:, i] == 0)) for i, key in enumerate(KPI_KEYS)},
    }
    path = assert_v2_write_path(SCALES_PATH)
    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _step_rewards(tr: dict, reward_set: str, scales: dict[str, float]) -> np.ndarray:
    use_rq, use_protect, use_ri, use_rc = SETS[reward_set]
    comp = np.asarray(tr["components"], dtype=np.float64)
    rq = comp[:, 0]
    if not use_rq and not use_protect:
        rq = np.zeros_like(rq)
    elif not use_protect:
        rq = np.maximum(rq, 0.0)
    elif not use_rq:
        rq = np.minimum(rq, 0.0)
    weights = np.asarray([CFG.reward.kpi_weights[k] for k in KPI_KEYS])
    denominators = np.asarray([scales[k] for k in KPI_KEYS])
    ri = (tr["dk"] / denominators * weights).sum(axis=1) if use_ri else 0.0
    rc = comp[:, 1] if use_rc else 0.0
    cost = comp[:, 2]
    rewards = CFG.reward.alpha * rq + CFG.reward.beta * ri + CFG.reward.gamma * rc - cost
    # STOP 行 components/dk 均为零，因此终止奖励保持 0。
    return np.asarray(rewards, dtype=np.float32)


def apply(reward_set: str, allow_partial: bool, overwrite: bool,
          cache_root=V2_CACHE_ROOT) -> None:
    if reward_set not in SETS:
        raise SystemExit(f"未知回报集 {reward_set}")
    if not SCALES_PATH.exists():
        raise SystemExit("缺 kpi_scales.json；先运行 calibrate")
    with SCALES_PATH.open(encoding="utf-8") as f:
        scale_meta = json.load(f)
    manifest = load_manifest()
    if scale_meta["manifest_hash"] != manifest_hash(manifest):
        raise SystemExit("KPI 标定文件与当前 manifest 不一致")
    cache_root = assert_v2_write_path(cache_root)
    root = assert_v2_write_path(cache_root / f"rtg_{reward_set}")
    root.mkdir(parents=True, exist_ok=True)
    index, missing = {}, []
    for name in manifest["train"] + manifest["val"]:
        try:
            trajectories = _load_tree(name, cache_root)
        except FileNotFoundError:
            missing.append(name)
            continue
        path = assert_v2_write_path(root / f"{name}.pkl")
        if path.exists() and not overwrite:
            with path.open("rb") as f:
                existing = pickle.load(f)
            index[name] = [{
                "policy": x["policy"], "steps": len(x["rtg"]),
                "total_reward": x["total_reward"]} for x in existing]
            continue
        output = []
        for tr in trajectories:
            rewards = _step_rewards(tr, reward_set, scale_meta["scales"])
            output.append({
                "policy": tr["policy"], "variant": tr.get("variant", 0),
                "rewards": rewards,
                "rtg": np.cumsum(rewards[::-1])[::-1].astype(np.float32),
                "total_reward": float(rewards.sum()),
            })
        with path.open("wb") as f:
            pickle.dump(output, f, protocol=4)
        index[name] = [{
            "policy": x["policy"], "steps": len(x["rtg"]),
            "total_reward": x["total_reward"]} for x in output]
    if missing and not allow_partial:
        raise SystemExit(f"{len(missing)} 棵缺 cache，未写完整 index：{missing[:5]}")
    meta = {
        "schema_version": 2, "dataset_version": CFG.data.v2_version,
        "manifest_hash": manifest_hash(manifest), "set": reward_set,
        "switches": SETS[reward_set], "alpha": CFG.reward.alpha,
        "beta": CFG.reward.beta, "gamma": CFG.reward.gamma,
        "kpi_weights": CFG.reward.kpi_weights,
        "kpi_scales": scale_meta["scales"],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (root / "index.json").open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "trees": index, "missing": missing},
                  f, ensure_ascii=False, indent=2)
    print(f"写入 {root}：{len(index)} 棵，缺 {len(missing)} 棵")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--quantile", type=float, default=0.95)
    cal.add_argument("--allow-partial", action="store_true")
    app = sub.add_parser("apply")
    app.add_argument("--set", choices=tuple(SETS), default="full")
    app.add_argument("--allow-partial", action="store_true")
    app.add_argument("--overwrite", action="store_true")
    app.add_argument("--cache-root", default=None,
                     help="可选并列v2缓存目录；相对 experiments/")
    args = ap.parse_args()
    if args.command == "calibrate":
        calibrate(args.quantile, args.allow_partial)
    else:
        cache_root = V2_CACHE_ROOT
        if args.cache_root:
            from pathlib import Path
            cache_root = Path(args.cache_root)
            if not cache_root.is_absolute():
                cache_root = Path(CFG.data.work_root) / cache_root
            cache_root = cache_root.resolve()
        apply(args.set, args.allow_partial, args.overwrite, cache_root)


if __name__ == "__main__":
    main()
