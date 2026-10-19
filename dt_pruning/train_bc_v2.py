"""使用 schema=2 状态与同一轨迹训练 BC 公平基线。"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from ablation_v2 import mask_graph_features
from config import CFG
from dt_model import BCPolicy, map_labels
from encoder import BRANCH_FEAT_DIM_V2, GLOBAL_FEAT_DIM_V2, BranchGraphEncoder
from graph_cache_v2 import cached_to_data, load_cache, load_split
from v2_manifest import load_manifest, manifest_hash


def flatten(trajs):
    return [(graph, int(label)) for tr in trajs
            for graph, label in zip(tr["graphs"], tr["labels"])]


def make_batch(samples, device, use_rule_features=True, use_kpi_features=True):
    graphs = Batch.from_data_list([cached_to_data(x[0]) for x in samples]).to(device)
    mask_graph_features(
        graphs, use_rule_features=use_rule_features,
        use_kpi_features=use_kpi_features)
    labels = torch.tensor([x[1] for x in samples], dtype=torch.long, device=device)
    return graphs, labels


def build_models(device):
    m = CFG.model
    encoder = BranchGraphEncoder(
        in_dim=BRANCH_FEAT_DIM_V2, global_dim=GLOBAL_FEAT_DIM_V2,
        hidden=m.gnn_hidden, heads=m.gnn_heads, layers=m.gnn_layers,
        state_dim=m.state_dim, dropout=m.dropout).to(device)
    policy = BCPolicy(
        state_dim=m.state_dim, node_dim=encoder.node_dim,
        hidden_dim=m.hidden_dim, dropout=m.dropout).to(device)
    return encoder, policy


@torch.no_grad()
def evaluate(encoder, policy, samples, batch_size, device,
             use_rule_features=True, use_kpi_features=True):
    encoder.eval(); policy.eval()
    loss_sum = hit = stop_hit = count = 0.0
    for pos in range(0, len(samples), batch_size):
        batch, labels = make_batch(
            samples[pos:pos + batch_size], device,
            use_rule_features, use_kpi_features)
        state, nodes, legal = encoder(batch)
        logits = policy(state, nodes, legal)
        mapped = map_labels(labels, nodes.size(1))
        loss_sum += F.cross_entropy(logits, mapped, reduction="sum").item()
        pred = logits.argmax(-1)
        hit += (pred == mapped).sum().item()
        stop_hit += ((pred == nodes.size(1)) == (labels == -1)).sum().item()
        count += len(labels)
    return {"loss": loss_sum / max(count, 1), "acc1": hit / max(count, 1),
            "stop_acc": stop_hit / max(count, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--policies", default="gardening,topk,kpi")
    ap.add_argument("--no-rule-features", action="store_true")
    ap.add_argument("--no-kpi-features", action="store_true")
    ap.add_argument("--max-trees", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", default="checkpoints_bc_v2")
    args = ap.parse_args()
    if "_v2" not in os.path.basename(args.out):
        raise SystemExit("v2 checkpoint 目录名必须含 _v2")
    torch.manual_seed(CFG.train.seed)
    rng = np.random.default_rng(CFG.train.seed)
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    split = load_split()
    train_names = split["train"][:args.max_trees] if args.max_trees else split["train"]
    n_val = max(1, (args.max_trees or 0) // 2) if args.max_trees else len(split["val"])
    val_names = split["val"][:n_val]
    policies = tuple(x.strip() for x in args.policies.split(",") if x.strip())
    train_samples = flatten(load_cache(train_names, policies))
    val_samples = flatten(load_cache(val_names, ("gardening",)))
    if not train_samples or not val_samples:
        raise SystemExit("v2 cache/rtg_full 为空")

    encoder, policy = build_models(device)
    params = list(encoder.parameters()) + list(policy.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=CFG.train.weight_decay)
    batches = math.ceil(len(train_samples) / args.batch_size)
    if args.max_batches:
        batches = min(batches, args.max_batches)
    total = max(batches * args.epochs, 1)
    warmup = max(int(total * CFG.train.warmup_ratio), 1)

    def schedule(step):
        if step < warmup:
            return step / warmup
        progress = min((step - warmup) / max(total - warmup, 1), 1.0)
        return CFG.train.min_lr_ratio + (1 - CFG.train.min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    out_dir = os.path.join(CFG.data.work_root, args.out)
    os.makedirs(out_dir, exist_ok=True)
    manifest = load_manifest()
    meta = {
        "kind": "bc_v2", "schema_version": 2,
        "manifest_hash": manifest_hash(manifest), "policies": list(policies),
        "use_rule_features": not args.no_rule_features,
        "use_kpi_features": not args.no_kpi_features,
        "feature_dim": BRANCH_FEAT_DIM_V2, "global_dim": GLOBAL_FEAT_DIM_V2,
        "train_trees": train_names, "val_trees": val_names,
    }
    log, best, started = [], float("inf"), time.time()
    for epoch in range(1, args.epochs + 1):
        encoder.train(); policy.train()
        order = rng.permutation(len(train_samples))
        loss_sum = count = 0
        for b in range(batches):
            samples = [train_samples[i] for i in order[
                b * args.batch_size:(b + 1) * args.batch_size]]
            graph, labels = make_batch(
                samples, device, not args.no_rule_features,
                not args.no_kpi_features)
            state, nodes, legal = encoder(graph)
            logits = policy(state, nodes, legal)
            mapped = map_labels(labels, nodes.size(1))
            loss = F.cross_entropy(logits, mapped)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CFG.train.grad_clip)
            optimizer.step(); scheduler.step()
            loss_sum += loss.item() * len(samples); count += len(samples)
        val = evaluate(
            encoder, policy, val_samples, args.batch_size, device,
            not args.no_rule_features, not args.no_kpi_features)
        rec = {"epoch": epoch, "train_loss": loss_sum / max(count, 1), "val": val,
               "minutes": (time.time() - started) / 60}
        log.append(rec)
        checkpoint = {"kind": "bc_v2", "encoder": encoder.state_dict(),
                      "model": policy.state_dict(), "meta": meta,
                      "epoch": epoch, "val": val}
        torch.save(checkpoint, os.path.join(out_dir, "bc_last.pt"))
        if val["loss"] < best:
            best = val["loss"]
            torch.save(checkpoint, os.path.join(out_dir, "bc_best.pt"))
        with open(os.path.join(out_dir, "bc_log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"epoch {epoch}/{args.epochs} train={rec['train_loss']:.4f} "
              f"val={val['loss']:.4f} acc={val['acc1']:.3f} stop={val['stop_acc']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
