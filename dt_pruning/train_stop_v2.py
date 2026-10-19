"""均衡微调 v2 Stop Head，不改变编码器、DT骨干或Branch/ΔK/净收益头。"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from config import CFG
from dt_model import PAD_LABEL, map_labels
from graph_cache_v2 import cached_to_data, load_cache, load_split
from train_v2 import build_models_from_meta
from v2_manifest import load_manifest, manifest_hash


def make_target_batch(trajs, jobs, context_len, rtg_scale, device):
    """每个窗口恰好终止于指定监督步；只对该步训练 STOP。"""
    graphs, labels, rtg, timesteps, mask, target_pos = [], [], [], [], [], []
    for row, (traj_idx, target) in enumerate(jobs):
        tr = trajs[int(traj_idx)]
        start = max(0, int(target) - context_len + 1)
        end = int(target) + 1
        length, pad = end - start, context_len - (end - start)
        graphs.extend(tr["graphs"][start:end] + [tr["graphs"][end - 1]] * pad)
        labels.append(list(tr["labels"][start:end]) + [PAD_LABEL] * pad)
        rtg.append(list(tr["rtg"][start:end] / rtg_scale) + [0.0] * pad)
        timesteps.append(list(range(start, end)) + [0] * pad)
        mask.append([1.0] * length + [0.0] * pad)
        target_pos.append(row * context_len + length - 1)
    return (
        Batch.from_data_list([cached_to_data(g) for g in graphs]).to(device),
        torch.tensor(labels, dtype=torch.long, device=device),
        torch.tensor(rtg, dtype=torch.float32, device=device),
        torch.tensor(timesteps, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
        torch.tensor(target_pos, dtype=torch.long, device=device),
    )


def stop_logits(encoder, model, batch):
    graph, labels, rtg, timesteps, mask, target_pos = batch
    batch_size, context_len = labels.shape
    with torch.no_grad():
        state, nodes, legal = encoder(graph)
        mapped = map_labels(labels.reshape(-1), nodes.size(1))
        action_features = model.action_features(nodes, mapped)
        ctx = model(
            state.view(batch_size, context_len, -1),
            action_features.view(batch_size, context_len, -1),
            rtg.unsqueeze(-1), timesteps, mask)
        # Stop Head 需要冻结的最佳候选表示与净收益；只让其自身 MLP 保留梯度。
        joint = (model.head.node_proj(nodes)
                 + model.head.ctx_proj(ctx.reshape(-1, ctx.size(-1))).unsqueeze(1))
        net_gain = model.head.net_gain(joint).squeeze(-1).masked_fill(
            ~legal, -1e9)
        best_idx = net_gain.argmax(dim=-1)
        rows = torch.arange(len(best_idx), device=nodes.device)
        stop_input = torch.cat([
            ctx.reshape(-1, ctx.size(-1)),
            joint[rows, best_idx],
            net_gain[rows, best_idx].unsqueeze(-1),
        ], dim=-1)
    return model.head.stop(stop_input[target_pos]).squeeze(-1)


def balanced_jobs(trajs, batch_size, rng):
    half = batch_size // 2
    positive = rng.choice(len(trajs), size=half, replace=True)
    negative_pool = [i for i, tr in enumerate(trajs) if len(tr["labels"]) >= 2]
    negative = rng.choice(negative_pool, size=batch_size - half, replace=True)
    jobs = [(int(i), len(trajs[int(i)]["labels"]) - 1) for i in positive]
    jobs += [(int(i), int(rng.integers(0, len(trajs[int(i)]["labels"]) - 1)))
             for i in negative]
    targets = torch.tensor([1.0] * half + [0.0] * (batch_size - half))
    order = rng.permutation(batch_size)
    return [jobs[i] for i in order], targets[order]


def validation_jobs(trajs):
    jobs, targets = [], []
    for i, tr in enumerate(trajs):
        total = len(tr["labels"])
        jobs.append((i, total - 1)); targets.append(1.0)
        if total >= 2:
            candidates = sorted(set(int(round((total - 2) * q))
                                    for q in (0.25, 0.5, 0.75)))
            jobs.extend((i, t) for t in candidates)
            targets.extend([0.0] * len(candidates))
    return jobs, torch.tensor(targets)


def metrics(logits, targets):
    pred = logits >= 0
    pos, neg = targets == 1, targets == 0
    sensitivity = (pred[pos].float().mean().item() if pos.any() else 0.0)
    specificity = ((~pred[neg]).float().mean().item() if neg.any() else 0.0)
    pos_loss = F.binary_cross_entropy_with_logits(
        logits[pos], targets[pos]) if pos.any() else logits.sum() * 0
    neg_loss = F.binary_cross_entropy_with_logits(
        logits[neg], targets[neg]) if neg.any() else logits.sum() * 0
    return {
        "balanced_loss": float((0.5 * (pos_loss + neg_loss)).item()),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_acc": 0.5 * (sensitivity + specificity),
        "mean_logit_pos": float(logits[pos].mean().item()) if pos.any() else None,
        "mean_logit_neg": float(logits[neg].mean().item()) if neg.any() else None,
    }


@torch.no_grad()
def validate(encoder, model, trajs, context_len, rtg_scale, batch_size, device):
    encoder.eval(); model.eval()
    jobs, targets = validation_jobs(trajs)
    outputs = []
    for pos in range(0, len(jobs), batch_size):
        chunk = jobs[pos:pos + batch_size]
        outputs.append(stop_logits(
            encoder, model,
            make_target_batch(trajs, chunk, context_len, rtg_scale, device)).cpu())
    return metrics(torch.cat(outputs), targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batches-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="checkpoints_dt_stop_v2")
    args = ap.parse_args()
    if args.batch_size < 2:
        raise SystemExit("--batch-size 至少为2")
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(
        CFG.data.work_root, args.checkpoint)
    source = torch.load(path, map_location=device, weights_only=False)
    if source.get("kind") != "dt_v2":
        raise SystemExit("--checkpoint 必须为 dt_v2")
    meta = dict(source["meta"])
    current_hash = manifest_hash(load_manifest())
    if meta.get("manifest_hash") != current_hash:
        raise SystemExit("checkpoint 与当前 manifest 不一致")
    encoder, model = build_models_from_meta(device, meta)
    encoder.load_state_dict(source["encoder"])
    model.load_state_dict(source["model"])
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.stop.parameters():
        parameter.requires_grad = True
    trainable = list(model.head.stop.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=CFG.train.weight_decay)

    split = load_split()
    policies = tuple(meta["policies"])
    train_trajs = load_cache(split["train"], policies)
    val_trajs = load_cache(split["val"], ("gardening",))
    context_len, rtg_scale = meta["context_len"], meta["rtg_scale"]
    rng = np.random.default_rng(CFG.train.seed)
    out_dir = os.path.join(CFG.data.work_root, args.out)
    os.makedirs(out_dir, exist_ok=True)
    log, best, started = [], float("inf"), time.time()
    for epoch in range(1, args.epochs + 1):
        model.head.stop.train()
        loss_sum = 0.0
        for _ in range(args.batches_per_epoch):
            jobs, targets = balanced_jobs(train_trajs, args.batch_size, rng)
            targets = targets.to(device)
            logits = stop_logits(
                encoder, model,
                make_target_batch(
                    train_trajs, jobs, context_len, rtg_scale, device))
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, CFG.train.grad_clip)
            optimizer.step()
            loss_sum += loss.item()
        val = validate(
            encoder, model, val_trajs, context_len, rtg_scale,
            args.batch_size, device)
        rec = {"epoch": epoch, "train_loss": loss_sum / args.batches_per_epoch,
               "val": val, "minutes": (time.time() - started) / 60}
        log.append(rec)
        tuned_meta = {
            **meta, "stop_finetune": {
                "source": args.checkpoint, "balanced_windows": True,
                "epochs": args.epochs, "selection": "val_balanced_loss"}}
        checkpoint = {
            **source, "encoder": encoder.state_dict(), "model": model.state_dict(),
            "epoch": epoch, "meta": tuned_meta, "val_stop": val}
        torch.save(checkpoint, os.path.join(out_dir, "dt_stop_last.pt"))
        if val["balanced_loss"] < best:
            best = val["balanced_loss"]
            torch.save(checkpoint, os.path.join(out_dir, "dt_stop_best.pt"))
        with open(os.path.join(out_dir, "stop_log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(
            f"epoch {epoch}/{args.epochs} train={rec['train_loss']:.4f} "
            f"val={val['balanced_loss']:.4f} sens={val['sensitivity']:.3f} "
            f"spec={val['specificity']:.3f} bacc={val['balanced_acc']:.3f}",
            flush=True)


if __name__ == "__main__":
    main()
