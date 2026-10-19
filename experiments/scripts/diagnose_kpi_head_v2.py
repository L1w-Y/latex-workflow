"""诊断现有v2 KPI辅助头：预测基线与共享层梯度，不修改模型。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dt_pruning"))

from config import CFG  # noqa: E402
from graph_cache_v2 import load_cache, load_split  # noqa: E402
from kpi_scales_v2 import activate_v2_kpi_scales  # noqa: E402
from train_v2 import (WindowSamplerV2, build_models_from_meta,  # noqa: E402
                      forward_batch)

KPI = ("hierarchy", "crowding", "balance", "light", "potential", "stability")


def grad_norm(loss, params):
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum(
        (g.detach() ** 2).sum() for g in grads if g is not None)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_dt_v2/dt_best.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="results/dev/kpi_redesign_v2/diagnosis_before.json")
    args = ap.parse_args()
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    path = Path(args.checkpoint)
    if not path.is_absolute():
        path = Path(CFG.data.work_root) / path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    scales = activate_v2_kpi_scales()
    dk_scales = np.asarray([scales[k] for k in KPI], dtype=np.float32)
    normalized = str(meta.get("kpi_target", "")).startswith(
        "delta_over_train_q95")
    target_scales = dk_scales if normalized else np.ones(6, dtype=np.float32)
    efficiency_ref = meta.get("kpi_efficiency_ref")
    encoder, model = build_models_from_meta(device, meta)
    encoder.load_state_dict(ckpt["encoder"])
    model.load_state_dict(ckpt["model"])

    split = load_split()
    policies = tuple(meta["policies"])
    train = load_cache(split["train"], policies, meta.get("reward_set", "full"))
    val = load_cache(split["val"], ("gardening",), meta.get("reward_set", "full"))
    train_rows = []
    for tr in train:
        for t in np.flatnonzero(tr["dk_mask"] & (tr["labels"] >= 0)):
            target = tr["dk"][t] / target_scales
            if efficiency_ref is not None:
                ratio = max(
                    float(tr["graphs"][t + 1]["g"][0])
                    - float(tr["graphs"][t]["g"][0]), 1e-8)
                target = target * float(efficiency_ref) / ratio
            train_rows.append(target)
    train_dk = np.asarray(train_rows)
    train_mean = train_dk.mean(axis=0)

    sampler = WindowSamplerV2(
        train, meta["context_len"], meta["rtg_scale"], CFG.train.seed, 0.5,
        dk_scales=target_scales, kpi_efficiency_ref=efficiency_ref)
    encoder.train(); model.train()
    batch = sampler.sample(args.batch_size, device)
    outputs = forward_batch(encoder, model, batch)
    branch, stop, dk_pred, _, labels, dk, dk_mask, _ = outputs
    valid = labels != -100
    branch_mask = valid & (labels >= 0)
    aux_mask = valid & dk_mask & (labels >= 0)
    stop_target = (labels == -1).float()
    branch_loss = F.cross_entropy(branch[branch_mask], labels[branch_mask])
    stop_loss = F.binary_cross_entropy_with_logits(
        stop[valid], stop_target[valid],
        pos_weight=stop.new_tensor(meta.get("stop_pos_weight", 8.0)))
    kpi_loss = F.smooth_l1_loss(dk_pred[aux_mask], dk[aux_mask])
    shared = list(encoder.parameters()) + list(model.head.node_proj.parameters()) \
        + list(model.head.ctx_proj.parameters())
    gradients = {
        "branch": grad_norm(branch_loss, shared),
        "stop": grad_norm(stop_loss, shared),
        "kpi_raw": grad_norm(kpi_loss, shared),
    }
    gradients["weighted_kpi"] = gradients["kpi_raw"] * meta.get("lambda_kpi", 0.2)
    gradients["weighted_kpi_over_branch"] = (
        gradients["weighted_kpi"] / max(gradients["branch"], 1e-12))

    encoder.eval(); model.eval()
    val_sampler = WindowSamplerV2(
        val, meta["context_len"], meta["rtg_scale"], 0, 1.0,
        dk_scales=target_scales, kpi_efficiency_ref=efficiency_ref)
    jobs = [(i, s) for i, tr in enumerate(val_sampler.trajs)
            for s in range(0, max(len(tr["labels"]) - 1, 1), val_sampler.K)]
    pred_rows, true_rows = [], []
    with torch.no_grad():
        for pos in range(0, len(jobs), args.batch_size):
            chunk = jobs[pos:pos + args.batch_size]
            out = forward_batch(
                encoder, model, val_sampler.sample(
                    len(chunk), device,
                    [x[0] for x in chunk], [x[1] for x in chunk]))
            mask = (out[4] >= 0) & out[6]
            pred_rows.append(out[2][mask].cpu().numpy())
            true_rows.append(out[5][mask].cpu().numpy())
    pred = np.concatenate(pred_rows)
    true = np.concatenate(true_rows)
    dimensions = {}
    for i, key in enumerate(KPI):
        mae = float(np.mean(np.abs(pred[:, i] - true[:, i])))
        baseline_mae = float(np.mean(np.abs(train_mean[i] - true[:, i])))
        denom = float(np.sum((true[:, i] - true[:, i].mean()) ** 2))
        r2 = 1.0 - float(np.sum((pred[:, i] - true[:, i]) ** 2)) / max(denom, 1e-12)
        rho = float(spearmanr(pred[:, i], true[:, i]).statistic)
        dimensions[key] = {
            "mae": mae, "mean_baseline_mae": baseline_mae,
            "mae_improvement": 1.0 - mae / max(baseline_mae, 1e-12),
            "r2": r2, "spearman": rho,
        }
    report = {
        "checkpoint": str(path), "n_val_actions": int(len(true)),
        "losses": {
            "branch": float(branch_loss.item()), "stop": float(stop_loss.item()),
            "kpi": float(kpi_loss.item()),
            "weighted_kpi": float(
                kpi_loss.item() * meta.get("lambda_kpi", 0.2)),
        },
        "shared_gradient_norms": gradients,
        "dimensions": dimensions,
        "mean_mae_improvement": float(np.mean(
            [x["mae_improvement"] for x in dimensions.values()])),
        "mean_spearman": float(np.nanmean(
            [x["spearman"] for x in dimensions.values()])),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(CFG.data.work_root) / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
