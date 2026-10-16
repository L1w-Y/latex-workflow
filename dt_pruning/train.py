"""DT + 分支级图编码器端到端训练（论文 sec:model:dt 口径）。

要点：
- 训练数据读 graph_cache 图缓存（整体驻留内存）；先运行 graph_cache.py
- 每条轨迹随机采样长度 K 的连续窗口作为一个样本（标准 DT 做法）
- 动作 = 当前树分枝列表中的位置或 STOP；action token 用被剪分枝的向量
- 损失为有效位置交叉熵；专家轨迹按 expert_weight 上调采样占比
- 每 eval_every 个 epoch 在验证树专家轨迹上统计 top-1/top-5，按 top-1 存最优
- --init-from 可用 BC checkpoint 初始化编码器；--policies random 时禁止该初始化

用法：
    python train.py --epochs 60 --context-len 16 --batch-size 8 --init-from checkpoints_bc/bc_best.pt
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
from dt_model import PAD_LABEL, DecisionTransformer, map_labels
from encoder import BranchGraphEncoder
from graph_cache import cached_to_data, load_cache, load_val_split


class WindowSampler:
    """每批采样 B 条轨迹 × 长度 K 的连续窗口（短轨迹用最后状态补齐，mask=0）。"""

    def __init__(self, trajs: list[dict], context_len: int, rtg_scale: float,
                 seed: int = 42, expert_weight: float = 0.0):
        self.trajs = [t for t in trajs if len(t["labels"]) >= 1]
        self.expert_idx = [i for i, t in enumerate(self.trajs) if t["policy"] == "expert"]
        self.other_idx = [i for i, t in enumerate(self.trajs) if t["policy"] != "expert"]
        self.expert_weight = expert_weight if self.expert_idx else 0.0
        self.K, self.rtg_scale = context_len, rtg_scale
        self.rng = np.random.default_rng(seed)

    def _pick(self, B: int) -> np.ndarray:
        if self.expert_weight <= 0.0 or not self.other_idx:
            return self.rng.choice(len(self.trajs), size=B, replace=True)
        n_exp = int(round(B * self.expert_weight))
        return np.concatenate([self.rng.choice(self.expert_idx, size=n_exp, replace=True),
                               self.rng.choice(self.other_idx, size=B - n_exp, replace=True)])

    def sample(self, B: int, device, picks=None, starts=None):
        K = self.K
        picks = self._pick(B) if picks is None else picks
        graphs, labels, rtg, ts, mask = [], [], [], [], []
        for k, i in enumerate(picks):
            tr = self.trajs[int(i)]
            T = len(tr["labels"])
            if starts is not None:
                start = int(starts[k])
            else:
                start = int(self.rng.integers(0, T - K + 1)) if T > K else 0
            end = min(start + K, T)
            L, pad = end - start, K - (end - start)
            graphs.extend(tr["graphs"][start:end] + [tr["graphs"][end - 1]] * pad)
            labels.append(list(tr["labels"][start:end]) + [PAD_LABEL] * pad)
            rtg.append(list(tr["rtg"][start:end] / self.rtg_scale) + [0.0] * pad)
            ts.append(list(range(start, end)) + [0] * pad)
            mask.append([1] * L + [0] * pad)
        is_exp = torch.tensor([self.trajs[int(i)]["policy"] == "expert" for i in picks],
                              device=device)
        return (Batch.from_data_list([cached_to_data(g) for g in graphs]).to(device),
                torch.tensor(labels, dtype=torch.long, device=device),
                torch.tensor(rtg, dtype=torch.float32, device=device),
                torch.tensor(ts, dtype=torch.long, device=device),
                torch.tensor(mask, dtype=torch.float32, device=device),
                is_exp)


def build_models(device, max_timestep: int = 1024):
    m = CFG.model
    encoder = BranchGraphEncoder(hidden=m.gnn_hidden, heads=m.gnn_heads,
                                 layers=m.gnn_layers, state_dim=m.state_dim,
                                 dropout=m.dropout).to(device)
    model = DecisionTransformer(state_dim=m.state_dim, node_dim=encoder.node_dim,
                                hidden_dim=m.hidden_dim, n_layer=m.n_layer,
                                n_head=m.n_head, dropout=m.dropout,
                                max_timestep=max_timestep).to(device)
    return encoder, model


def forward_batch(encoder, model, batch):
    """返回 (logits [B*K, N+1], labels [B*K]（已映射）, is_exp [B])。"""
    graph, labels, rtg, ts, mask, is_exp = batch
    B, K = labels.shape
    state, nodes, legal = encoder(graph)
    lab = map_labels(labels.view(-1), nodes.size(1))
    act = model.action_features(nodes, lab)
    ctx = model(state.view(B, K, -1), act.view(B, K, -1), rtg.unsqueeze(-1), ts, mask)
    logits = model.head(ctx.reshape(B * K, -1), nodes, legal)
    return logits, lab, is_exp


def accuracy(logits, lab, is_exp, K):
    """专家窗口有效位置的 (top1 命中, top5 命中, 位置数)。"""
    exp_pos = is_exp.repeat_interleave(K) & (lab != PAD_LABEL)
    if not exp_pos.any():
        return 0, 0, 0
    lg, lb = logits[exp_pos], lab[exp_pos]
    top5 = lg.topk(min(5, lg.size(1)), dim=-1).indices
    return ((top5[:, 0] == lb).sum().item(),
            (top5 == lb.unsqueeze(-1)).any(-1).sum().item(), lb.numel())


def all_accuracy(logits, lab):
    """全部有效位置的 (top1 命中, top5 命中, 位置数)，不区分轨迹来源。"""
    valid = lab != PAD_LABEL
    if not valid.any():
        return 0, 0, 0
    lg, lb = logits[valid], lab[valid]
    top5 = lg.topk(min(5, lg.size(1)), dim=-1).indices
    return ((top5[:, 0] == lb).sum().item(),
            (top5 == lb.unsqueeze(-1)).any(-1).sum().item(), lb.numel())


@torch.no_grad()
def validate(encoder, model, val_sampler, batch_size, n_batches, device):
    """验证树专家轨迹：固定窗口集合（每条轨迹按步长 K 平铺），结果可复现。"""
    encoder.eval(); model.eval()
    K = val_sampler.K
    jobs = [(i, s) for i in val_sampler.expert_idx
            for s in range(0, max(len(val_sampler.trajs[i]["labels"]) - 1, 1), K)]
    jobs = jobs[:batch_size * n_batches] if n_batches else jobs
    h1 = h5 = n = 0; loss_sum = 0.0
    for b in range(0, len(jobs), batch_size):
        chunk = jobs[b:b + batch_size]
        batch = val_sampler.sample(len(chunk), device, picks=[j[0] for j in chunk],
                                   starts=[j[1] for j in chunk])
        logits, lab, is_exp = forward_batch(encoder, model, batch)
        valid = lab != PAD_LABEL
        loss_sum += F.cross_entropy(logits[valid], lab[valid], reduction="sum").item()
        a, c, d = accuracy(logits, lab, is_exp, K)
        h1 += a; h5 += c; n += d
    return {"loss": loss_sum / max(n, 1), "acc1": h1 / max(n, 1), "acc5": h5 / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--context-len", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=CFG.train.lr)
    ap.add_argument("--rtg-scale", type=float, default=1000.0)
    ap.add_argument("--batches-per-epoch", type=int, default=200)
    ap.add_argument("--expert-weight", type=float, default=0.5,
                    help="专家轨迹采样占比（数据中仅占 20%%）；不含专家轨迹时必须为 0")
    ap.add_argument("--policies", type=str, default="expert,random",
                    help="训练轨迹类型，逗号分隔；去园艺知识消融用 random")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--val-batches", type=int, default=None,
                    help="验证批数上限（默认用全部验证窗口）")
    ap.add_argument("--init-from", type=str, default=None,
                    help="用 BC checkpoint 初始化编码器")
    ap.add_argument("--max-trees", type=int, default=None, help="冒烟测试：只用前 N 棵训练树")
    ap.add_argument("--reward-set", type=str, default="full", help="回报集（graph_cache/rtg_<set>）")
    ap.add_argument("--out", type=str, default="checkpoints_dt")
    args = ap.parse_args()

    torch.manual_seed(CFG.train.seed)
    np.random.seed(CFG.train.seed)
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")

    policies = tuple(p.strip() for p in args.policies.split(",") if p.strip())
    if not policies or any(p not in ("expert", "random") for p in policies):
        raise SystemExit("--policies 只能包含 expert 和 random")
    if "expert" not in policies and args.init_from:
        raise SystemExit("去掉专家轨迹时不能从专家 BC checkpoint 初始化")
    if "expert" not in policies and args.expert_weight != 0:
        raise SystemExit("训练不含专家轨迹时 --expert-weight 必须为 0")

    split = load_val_split()
    train_names = split["train"][:args.max_trees] if args.max_trees else split["train"]
    val_names = split["val"][:max(1, args.max_trees // 3)] if args.max_trees else split["val"]
    t0 = time.time()
    train_trajs = load_cache(train_names, policies, reward_set=args.reward_set)
    val_trajs = load_cache(val_names, ("expert",), reward_set=args.reward_set)
    counts = {p: sum(t["policy"] == p for t in train_trajs) for p in ("expert", "random")}
    print(f"设备 {device}；训练树 {len(train_names)}（{len(train_trajs)} 条轨迹："
          f"专家 {counts['expert']}，随机 {counts['random']}），"
          f"验证树 {len(val_names)}；读缓存 {(time.time()-t0)/60:.1f} min", flush=True)
    if "expert" not in policies and counts["expert"]:
        raise SystemExit("已要求去掉专家轨迹，但训练集仍含专家轨迹")
    if not train_trajs:
        raise SystemExit("训练轨迹为空：请先运行 graph_cache.py")

    sampler = WindowSampler(train_trajs, args.context_len, args.rtg_scale,
                            seed=CFG.train.seed, expert_weight=args.expert_weight)
    val_sampler = WindowSampler(val_trajs, args.context_len, args.rtg_scale, seed=0)

    encoder, model = build_models(device)
    if args.init_from:
        path = args.init_from if os.path.isabs(args.init_from) \
            else os.path.join(CFG.data.work_root, args.init_from)
        encoder.load_state_dict(torch.load(path, map_location=device,
                                           weights_only=False)["encoder"])
        print("编码器初始化自", path)
    params = list(encoder.parameters()) + list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=CFG.train.weight_decay)
    total = args.batches_per_epoch * args.epochs
    warmup = max(int(total * CFG.train.warmup_ratio), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        p = min((step - warmup) / max(total - warmup, 1), 1.0)
        return CFG.train.min_lr_ratio + (1 - CFG.train.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    out_dir = os.path.join(CFG.data.work_root, args.out)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"kind": "dt", "state_dim": CFG.model.state_dim, "rtg_scale": args.rtg_scale,
            "context_len": args.context_len, "max_timestep": model.max_timestep,
            "policies": list(policies), "expert_weight": args.expert_weight,
            "init_from": args.init_from, "checkpoint_selection": "val_expert_acc1",
            "train_trees": len(train_names), "train_traj_counts": counts,
            "val_trees": val_names, "reward_set": args.reward_set}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    K = args.context_len
    log, best, t0 = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        encoder.train(); model.train()
        loss_sum = 0.0; h1 = h5 = n = 0; ah1 = ah5 = an = 0
        for _ in range(args.batches_per_epoch):
            logits, lab, is_exp = forward_batch(encoder, model,
                                                sampler.sample(args.batch_size, device))
            loss = F.cross_entropy(logits, lab, ignore_index=PAD_LABEL)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CFG.train.grad_clip)
            opt.step(); sched.step()
            loss_sum += loss.item()
            a, c, d = accuracy(logits.detach(), lab, is_exp, K)
            h1 += a; h5 += c; n += d
            aa, ac, ad = all_accuracy(logits.detach(), lab)
            ah1 += aa; ah5 += ac; an += ad
        rec = {"epoch": epoch, "loss": loss_sum / args.batches_per_epoch,
               "acc1_expert": h1 / max(n, 1), "acc5_expert": h5 / max(n, 1),
               "acc1_all": ah1 / max(an, 1), "acc5_all": ah5 / max(an, 1),
               "lr": sched.get_last_lr()[0], "minutes": (time.time() - t0) / 60}
        if "expert" in policies:
            acc_msg = (f"acc1_expert {rec['acc1_expert']:.3f} "
                       f"acc5 {rec['acc5_expert']:.3f}")
        else:
            acc_msg = (f"acc1 {rec['acc1_all']:.3f} acc5 {rec['acc5_all']:.3f}"
                       "（训练无专家轨迹，验证仍按专家动作选模）")
        msg = f"epoch {epoch}/{args.epochs}  loss {rec['loss']:.3f}  {acc_msg}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = validate(encoder, model, val_sampler, args.batch_size,
                           args.val_batches, device)
            rec["val"] = val
            msg += f" | val loss {val['loss']:.3f} acc1 {val['acc1']:.3f} acc5 {val['acc5']:.3f}"
            ckpt = {"kind": "dt", "encoder": encoder.state_dict(),
                    "model": model.state_dict(), "epoch": epoch, "meta": meta, "val": val}
            torch.save(ckpt, os.path.join(out_dir, "dt_last.pt"))
            if val["acc1"] > best:
                best = val["acc1"]
                torch.save(ckpt, os.path.join(out_dir, "dt_best.pt"))
        log.append(rec)
        print(msg + f" | {rec['minutes']:.1f} min", flush=True)
        with open(os.path.join(out_dir, "loss_log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)
    print(f"训练完成：最优验证 acc1 {best:.3f}，checkpoint 目录 {out_dir}")


if __name__ == "__main__":
    main()
