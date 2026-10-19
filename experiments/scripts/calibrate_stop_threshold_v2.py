"""在10棵选模树的全部轨迹状态上标定 Stop Head 概率阈值。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dt_pruning"))

from config import CFG  # noqa: E402
from graph_cache_v2 import load_cache, load_split  # noqa: E402
from train_stop_v2 import make_target_batch, stop_logits  # noqa: E402
from train_v2 import build_models_from_meta  # noqa: E402
from v2_manifest import load_manifest, manifest_hash  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(
        CFG.data.work_root, args.checkpoint)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    meta = checkpoint["meta"]
    if checkpoint.get("kind") != "dt_v2":
        raise SystemExit("只支持 dt_v2 checkpoint")
    if meta.get("manifest_hash") != manifest_hash(load_manifest()):
        raise SystemExit("checkpoint 与当前 manifest 不一致")
    encoder, model = build_models_from_meta(device, meta)
    encoder.load_state_dict(checkpoint["encoder"])
    model.load_state_dict(checkpoint["model"])
    encoder.eval(); model.eval()
    trajectories = load_cache(load_split()["val"], ("gardening",))
    jobs, targets = [], []
    for i, tr in enumerate(trajectories):
        total = len(tr["labels"])
        jobs.extend((i, t) for t in range(total))
        targets.extend([0] * (total - 1) + [1])
    outputs = []
    for pos in range(0, len(jobs), args.batch_size):
        chunk = jobs[pos:pos + args.batch_size]
        outputs.append(stop_logits(
            encoder, model,
            make_target_batch(
                trajectories, chunk, meta["context_len"], meta["rtg_scale"], device)
        ).cpu())
    logits = torch.cat(outputs).numpy()
    targets = np.asarray(targets, dtype=bool)
    candidates = np.unique(np.r_[logits, logits.min() - 1e-6, logits.max() + 1e-6])
    rows = []
    for threshold in candidates:
        pred = logits >= threshold
        sensitivity = float(pred[targets].mean())
        specificity = float((~pred[~targets]).mean())
        rows.append({
            "logit_threshold": float(threshold),
            "probability_threshold": float(1.0 / (1.0 + np.exp(-threshold))),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_acc": 0.5 * (sensitivity + specificity),
        })
    # 首要最大化 balanced accuracy；并列时优先灵敏度，再优先更高阈值。
    best = max(rows, key=lambda x: (
        x["balanced_acc"], x["sensitivity"], x["logit_threshold"]))
    result = {
        "checkpoint": path, "manifest_hash": meta["manifest_hash"],
        "n_trajectories": len(trajectories), "n_positive": int(targets.sum()),
        "n_negative": int((~targets).sum()), "best": best,
        "default_0_5": next(
            (row for row in rows if row["logit_threshold"] >= 0), rows[-1]),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = args.out or os.path.join(
        os.path.dirname(path), "stop_threshold_calibration.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("写入", out)


if __name__ == "__main__":
    main()
