"""单步行为克隆（BC）：编码器 + 分枝打分动作头拟合专家动作。

用途：①验证状态特征是否含可指导决策的信息（风险试水第二层①）；
②作为论文 BC 基线；③其编码器可作为 DT 训练的初始化（--init-from）。

每个 epoch 结束在验证树上统计 top-1 / top-5 准确率与 STOP 判断准确率，
按验证 top-1 保存最优 checkpoint。

用法：
    python train_bc.py --epochs 30 --batch-size 64
"""
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

from config import CFG
from dt_model import PAD_LABEL, BCPolicy, map_labels
from encoder import BranchGraphEncoder
from graph_cache import cached_to_data, load_cache, load_val_split


def flatten(trajs: list[dict]) -> list[tuple[dict, int]]:
    return [(g, int(a)) for tr in trajs for g, a in zip(tr["graphs"], tr["labels"])]


def make_batch(samples, device):
    batch = Batch.from_data_list([cached_to_data(g) for g, _ in samples]).to(device)
    labels = torch.tensor([a for _, a in samples], dtype=torch.long, device=device)
    return batch, labels


def build_models(device):
    m = CFG.model
    encoder = BranchGraphEncoder(hidden=m.gnn_hidden, heads=m.gnn_heads,
                                 layers=m.gnn_layers, state_dim=m.state_dim,
                                 dropout=m.dropout).to(device)
    policy = BCPolicy(state_dim=m.state_dim, node_dim=encoder.node_dim,
                      hidden_dim=m.hidden_dim, dropout=m.dropout).to(device)
    return encoder, policy


def step_metrics(logits, labels):
    """返回 (top1 命中, top5 命中, STOP 判断正确, 样本数)。"""
    n = logits.size(1) - 1
    top5 = logits.topk(min(5, logits.size(1)), dim=-1).indices
    top1 = top5[:, 0]
    hit1 = (top1 == labels).sum().item()
    hit5 = (top5 == labels.unsqueeze(-1)).any(-1).sum().item()
    stop_ok = ((top1 == n) == (labels == n)).sum().item()
    return hit1, hit5, stop_ok, labels.numel()


@torch.no_grad()
def evaluate(encoder, policy, samples, batch_size, device):
    encoder.eval(); policy.eval()
    tot = np.zeros(4); loss_sum = 0.0
    for i in range(0, len(samples), batch_size):
        batch, labels = make_batch(samples[i:i + batch_size], device)
        state, nodes, legal = encoder(batch)
        logits = policy(state, nodes, legal)
        labels = map_labels(labels, nodes.size(1))
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        tot += step_metrics(logits, labels)
    n = max(tot[3], 1)
    return {"loss": loss_sum / n, "acc1": tot[0] / n, "acc5": tot[1] / n,
            "stop_acc": tot[2] / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--policies", type=str, default="expert",
                    help="参与训练的轨迹类型，逗号分隔（默认只用专家）")
    ap.add_argument("--max-trees", type=int, default=None, help="冒烟测试：只用前 N 棵训练树")
    ap.add_argument("--max-batches", type=int, default=None, help="冒烟测试：每 epoch 最多批数")
    ap.add_argument("--out", type=str, default="checkpoints_bc")
    args = ap.parse_args()

    torch.manual_seed(CFG.train.seed)
    rng = np.random.default_rng(CFG.train.seed)
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")

    split = load_val_split()
    train_names = split["train"][:args.max_trees] if args.max_trees else split["train"]
    val_names = split["val"][:max(1, (args.max_trees or 0) // 3)] if args.max_trees else split["val"]
    policies = tuple(args.policies.split(","))
    train_s = flatten(load_cache(train_names, policies))
    val_s = flatten(load_cache(val_names, ("expert",)))
    print(f"设备 {device}；训练树 {len(train_names)}（{len(train_s)} 步），"
          f"验证树 {len(val_names)}（{len(val_s)} 步专家决策）")
    if not train_s:
        raise SystemExit("训练样本为空：请先运行 graph_cache.py")

    encoder, policy = build_models(device)
    params = list(encoder.parameters()) + list(policy.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=CFG.train.weight_decay)
    nb = math.ceil(len(train_s) / args.batch_size)
    if args.max_batches:
        nb = min(nb, args.max_batches)
    total = nb * args.epochs
    warmup = max(int(total * CFG.train.warmup_ratio), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        p = min((step - warmup) / max(total - warmup, 1), 1.0)
        return CFG.train.min_lr_ratio + (1 - CFG.train.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    out_dir = os.path.join(CFG.data.work_root, args.out)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"kind": "bc", "state_dim": CFG.model.state_dim, "policies": list(policies),
            "train_trees": len(train_names), "val_trees": val_names}
    log, best, t0 = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        encoder.train(); policy.train()
        order = rng.permutation(len(train_s))
        tot = np.zeros(4); loss_sum = 0.0
        for b in range(nb):
            samples = [train_s[i] for i in order[b * args.batch_size:(b + 1) * args.batch_size]]
            batch, labels = make_batch(samples, device)
            state, nodes, legal = encoder(batch)
            logits = policy(state, nodes, legal)
            labels = map_labels(labels, nodes.size(1))
            loss = F.cross_entropy(logits, labels, ignore_index=PAD_LABEL)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CFG.train.grad_clip)
            opt.step(); sched.step()
            loss_sum += loss.item() * labels.numel()
            tot += step_metrics(logits.detach(), labels)
        n = max(tot[3], 1)
        val = evaluate(encoder, policy, val_s, args.batch_size, device)
        rec = {"epoch": epoch, "train_loss": loss_sum / n, "train_acc1": tot[0] / n,
               "val": val, "lr": sched.get_last_lr()[0], "minutes": (time.time() - t0) / 60}
        log.append(rec)
        print(f"epoch {epoch}/{args.epochs}  train loss {rec['train_loss']:.3f} acc1 {rec['train_acc1']:.3f} | "
              f"val loss {val['loss']:.3f} acc1 {val['acc1']:.3f} acc5 {val['acc5']:.3f} "
              f"stop {val['stop_acc']:.3f} | {rec['minutes']:.1f} min", flush=True)
        ckpt = {"kind": "bc", "encoder": encoder.state_dict(), "model": policy.state_dict(),
                "epoch": epoch, "meta": meta, "val": val}
        torch.save(ckpt, os.path.join(out_dir, "bc_last.pt"))
        if val["acc1"] > best:
            best = val["acc1"]
            torch.save(ckpt, os.path.join(out_dir, "bc_best.pt"))
        with open(os.path.join(out_dir, "bc_log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)
    print(f"完成：最优验证 acc1 {best:.3f}，checkpoint 目录 {out_dir}")


if __name__ == "__main__":
    main()
