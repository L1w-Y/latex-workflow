"""用冻结模型自身 rollout 的 oracle 特征微调 Stop Head。"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
from train_v2 import build_models_from_meta
from v2_manifest import load_manifest, manifest_hash


def evaluate(head, x, y):
    with torch.no_grad():
        logits = head(x).squeeze(-1)
        pred = logits >= 0
        pos, neg = y == 1, y == 0
        sens = pred[pos].float().mean().item() if pos.any() else 0.0
        spec = (~pred[neg]).float().mean().item() if neg.any() else 0.0
        pos_loss = F.binary_cross_entropy_with_logits(logits[pos], y[pos])
        neg_loss = F.binary_cross_entropy_with_logits(logits[neg], y[neg])
    return {
        "balanced_loss": float((0.5 * (pos_loss + neg_loss)).item()),
        "sensitivity": sens, "specificity": spec,
        "balanced_acc": 0.5 * (sens + spec),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/stop_onpolicy_v2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps-per-epoch", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="checkpoints_dt_stop_onpolicy_v2")
    args = ap.parse_args()
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(
        CFG.data.work_root, args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    meta = dict(checkpoint["meta"])
    if meta.get("manifest_hash") != manifest_hash(load_manifest()):
        raise SystemExit("checkpoint 与当前 manifest 不一致")
    encoder, model = build_models_from_meta(device, meta)
    encoder.load_state_dict(checkpoint["encoder"])
    model.load_state_dict(checkpoint["model"])
    data_root = args.data if os.path.isabs(args.data) else os.path.join(
        CFG.data.work_root, args.data)
    with np.load(os.path.join(data_root, "train.npz")) as d:
        x_train = torch.from_numpy(d["x"]).to(device)
        y_train = torch.from_numpy(d["y"]).to(device)
    with np.load(os.path.join(data_root, "val.npz")) as d:
        x_val = torch.from_numpy(d["x"]).to(device)
        y_val = torch.from_numpy(d["y"]).to(device)
    pos_idx = torch.nonzero(y_train == 1, as_tuple=False).flatten()
    neg_idx = torch.nonzero(y_train == 0, as_tuple=False).flatten()
    if not len(pos_idx) or not len(neg_idx):
        raise SystemExit("on-policy 数据必须同时包含 STOP 与继续样本")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.stop.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(
        model.head.stop.parameters(), lr=args.lr,
        weight_decay=CFG.train.weight_decay)
    generator = torch.Generator(device=device).manual_seed(CFG.train.seed)
    half = args.batch_size // 2
    out_dir = os.path.join(CFG.data.work_root, args.out)
    os.makedirs(out_dir, exist_ok=True)
    best, log, started = float("inf"), [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.head.stop.train()
        loss_sum = 0.0
        for _ in range(args.steps_per_epoch):
            pi = pos_idx[torch.randint(
                len(pos_idx), (half,), generator=generator, device=device)]
            ni = neg_idx[torch.randint(
                len(neg_idx), (args.batch_size - half,),
                generator=generator, device=device)]
            idx = torch.cat([pi, ni])
            logits = model.head.stop(x_train[idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_train[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()
        model.head.stop.eval()
        val = evaluate(model.head.stop, x_val, y_val)
        rec = {"epoch": epoch, "train_loss": loss_sum / args.steps_per_epoch,
               "val": val, "minutes": (time.time() - started) / 60}
        log.append(rec)
        tuned_meta = {
            **meta, "stop_onpolicy_finetune": {
                "source": args.checkpoint, "data": args.data,
                "selection": "val_balanced_loss"}}
        output = {
            **checkpoint, "model": model.state_dict(), "meta": tuned_meta,
            "epoch": epoch, "val_stop_onpolicy": val}
        torch.save(output, os.path.join(out_dir, "dt_stop_onpolicy_last.pt"))
        if val["balanced_loss"] < best:
            best = val["balanced_loss"]
            torch.save(output, os.path.join(out_dir, "dt_stop_onpolicy_best.pt"))
        with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch {epoch}/{args.epochs} train={rec['train_loss']:.4f} "
                f"val={val['balanced_loss']:.4f} sens={val['sensitivity']:.3f} "
                f"spec={val['specificity']:.3f} bacc={val['balanced_acc']:.3f}",
                flush=True)


if __name__ == "__main__":
    main()
