"""评估KPI头在同一状态多个候选之间的真实排序能力。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dt_pruning"))

from config import CFG  # noqa: E402
from graph_cache_v2 import load_cache, load_split  # noqa: E402
from kpi_scales_v2 import activate_v2_kpi_scales  # noqa: E402
from train_v2 import (WindowSamplerV2, attach_candidate_targets,  # noqa: E402
                      build_models_from_meta, forward_batch)

KPI = ("hierarchy", "crowding", "balance", "light", "potential", "stability")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate-root", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    scales = activate_v2_kpi_scales()
    dk_scales = np.asarray([scales[k] for k in KPI], dtype=np.float32)
    encoder, model = build_models_from_meta(device, meta)
    encoder.load_state_dict(ckpt["encoder"])
    model.load_state_dict(ckpt["model"])
    encoder.eval()
    model.eval()

    val = load_cache(
        load_split()["val"], ("gardening",), meta.get("reward_set", "full"))
    candidate_root = Path(args.candidate_root)
    if not candidate_root.is_absolute():
        candidate_root = ROOT / candidate_root
    attach_candidate_targets(val, candidate_root)
    sampler = WindowSamplerV2(
        val, meta["context_len"], meta["rtg_scale"], 0, 1.0,
        dk_scales=dk_scales,
        kpi_efficiency_ref=meta.get("kpi_efficiency_ref"))
    jobs = [(i, s) for i, tr in enumerate(sampler.trajs)
            for s in range(0, max(len(tr["labels"]) - 1, 1), sampler.K)]

    pred_rows, true_rows = [], []
    state_rho, top1, random_top1 = [], [], []
    weights = model.head.kpi_weights.detach().cpu().numpy()
    with torch.no_grad():
        for pos in range(0, len(jobs), args.batch_size):
            chunk = jobs[pos:pos + args.batch_size]
            batch = sampler.sample(
                len(chunk), device,
                [x[0] for x in chunk], [x[1] for x in chunk])
            _, candidate = forward_batch(
                encoder, model, batch, return_candidate=True)
            pred, target, mask = candidate
            for i in range(len(pred)):
                valid = mask[i]
                n = int(valid.sum())
                if n < 2:
                    continue
                p = pred[i, valid].cpu().numpy()
                y = target[i, valid].cpu().numpy()
                pu, yu = p @ weights, y @ weights
                rho = spearmanr(pu, yu).statistic
                if np.isfinite(rho):
                    state_rho.append(float(rho))
                top1.append(int(np.argmax(pu) == np.argmax(yu)))
                random_top1.append(1.0 / n)
                pred_rows.append(p)
                true_rows.append(y)
    pred = np.concatenate(pred_rows)
    true = np.concatenate(true_rows)
    dimensions = {}
    for i, key in enumerate(KPI):
        mae = float(np.mean(np.abs(pred[:, i] - true[:, i])))
        baseline = float(np.mean(np.abs(true[:, i] - true[:, i].mean())))
        dimensions[key] = {
            "mae": mae,
            "mean_baseline_mae": baseline,
            "mae_improvement": 1.0 - mae / max(baseline, 1e-12),
            "spearman": float(spearmanr(pred[:, i], true[:, i]).statistic),
        }
    report = {
        "checkpoint": str(checkpoint),
        "n_candidate_states": len(top1),
        "n_candidates": int(len(true)),
        "candidate_utility_mean_state_spearman": float(np.mean(state_rho)),
        "candidate_utility_top1": float(np.mean(top1)),
        "candidate_utility_random_top1": float(np.mean(random_top1)),
        "candidate_utility_top1_lift": float(
            np.mean(top1) - np.mean(random_top1)),
        "dimensions": dimensions,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
