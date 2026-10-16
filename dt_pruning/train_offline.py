"""离线强化学习基线训练：CQL / BCQ / IQL（与 DT 同数据、同编码器、同初始化）。

数据：图缓存中训练树的全部轨迹（专家 + 随机），按 expert_weight 上调专家轨迹采样占比
（与 DT 一致）；转移 (s_t, a_t, r_t, s_{t+1}, done)，r_t 由剩余回报相邻两步之差得到，
每条轨迹最后一步为 STOP（终止）。奖励乘 reward_scale 后参与 TD 学习。
目标网络（编码器 + 各头）以 polyak 系数 tau 软更新。
按固定梯度步数训练，保存最后一步 checkpoint（离线 RL 通行做法），中途按间隔另存。

用法：
    python train_offline.py --algo cql --init-from checkpoints_bc/bc_best.pt --out checkpoints_cql
    python train_offline.py --algo bcq --init-from checkpoints_bc/bc_best.pt --out checkpoints_bcq
    python train_offline.py --algo iql --init-from checkpoints_bc/bc_best.pt --out checkpoints_iql
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from config import CFG
from dt_model import map_labels
from encoder import BranchGraphEncoder
from graph_cache import cached_to_data, load_cache, load_val_split
from offline_rl import ALGOS, OfflineAgent, compute_loss, polyak


class TransitionSampler:
    def __init__(self, trajs: list[dict], seed: int, expert_weight: float):
        self.trajs = [t for t in trajs if len(t["labels"]) >= 1]
        self.expert_idx = [i for i, t in enumerate(self.trajs) if t["policy"] == "expert"]
        self.other_idx = [i for i, t in enumerate(self.trajs) if t["policy"] != "expert"]
        self.expert_weight = expert_weight if self.expert_idx and self.other_idx else 0.0
        self.rng = np.random.default_rng(seed)

    def _pick(self, B: int) -> np.ndarray:
        if self.expert_weight <= 0.0:
            return self.rng.choice(len(self.trajs), size=B, replace=True)
        n_exp = int(round(B * self.expert_weight))
        return np.concatenate([self.rng.choice(self.expert_idx, size=n_exp, replace=True),
                               self.rng.choice(self.other_idx, size=B - n_exp, replace=True)])

    def build(self, picks, steps, device):
        g_s, g_n, labels, reward, done = [], [], [], [], []
        for i, t in zip(picks, steps):
            tr = self.trajs[int(i)]
            T = len(tr["labels"])
            last = t == T - 1
            g_s.append(tr["graphs"][t])
            g_n.append(tr["graphs"][t] if last else tr["graphs"][t + 1])
            labels.append(int(tr["labels"][t]))
            reward.append(float(tr["rtg"][t] - (0.0 if last else tr["rtg"][t + 1])))
            done.append(float(last))
        to_b = lambda gs: Batch.from_data_list([cached_to_data(g) for g in gs]).to(device)
        return (to_b(g_s), to_b(g_n),
                torch.tensor(labels, dtype=torch.long, device=device),
                torch.tensor(reward, dtype=torch.float32, device=device),
                torch.tensor(done, dtype=torch.float32, device=device))

    def sample(self, B: int, device):
        picks = self._pick(B)
        steps = [int(self.rng.integers(0, len(self.trajs[int(i)]["labels"]))) for i in picks]
        return self.build(picks, steps, device)


def build_models(device, algo: str, bcq_threshold: float = 0.3):
    m = CFG.model
    encoder = BranchGraphEncoder(hidden=m.gnn_hidden, heads=m.gnn_heads,
                                 layers=m.gnn_layers, state_dim=m.state_dim,
                                 dropout=m.dropout).to(device)
    agent = OfflineAgent(algo, state_dim=m.state_dim, node_dim=encoder.node_dim,
                         hidden_dim=m.hidden_dim, dropout=m.dropout,
                         bcq_threshold=bcq_threshold).to(device)
    return encoder, agent


def train_step(encoder, agent, enc_t, agent_t, batch, hp):
    g_s, g_n, labels, reward, done = batch
    out = encoder(g_s)
    lab = map_labels(labels, out[1].size(1))
    with torch.no_grad():
        nxt = enc_t(g_n)
        cur_t = enc_t(g_s) if agent.algo == "iql" else None
    return compute_loss(agent, agent_t, out, nxt, cur_t, lab, reward * hp["reward_scale"],
                        done, hp)


@torch.no_grad()
def validate(encoder, agent, val_sampler, batch_size, device):
    """验证树专家轨迹逐步：动作 argmax 与专家一致率（诊断用，不用于选模型）。"""
    encoder.eval(); agent.eval()
    jobs = [(i, t) for i in range(len(val_sampler.trajs))
            for t in range(len(val_sampler.trajs[i]["labels"]))]
    hit = n = 0
    for b in range(0, len(jobs), batch_size):
        chunk = jobs[b:b + batch_size]
        g_s, _, labels, _, _ = val_sampler.build([j[0] for j in chunk], [j[1] for j in chunk], device)
        state, nodes, legal = encoder(g_s)
        lab = map_labels(labels, nodes.size(1))
        hit += (agent(state, nodes, legal).argmax(-1) == lab).sum().item()
        n += lab.numel()
    encoder.train(); agent.train()
    return {"acc1_vs_expert": hit / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS, required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=CFG.train.lr)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--reward-scale", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    ap.add_argument("--expert-weight", type=float, default=0.5, help="专家轨迹采样占比（同 DT）")
    ap.add_argument("--cql-alpha", type=float, default=1.0)
    ap.add_argument("--bcq-threshold", type=float, default=0.3)
    ap.add_argument("--iql-tau", type=float, default=0.7, help="expectile")
    ap.add_argument("--iql-beta", type=float, default=3.0, help="优势加权温度")
    ap.add_argument("--init-from", type=str, default=None, help="用 BC checkpoint 初始化编码器")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--max-trees", type=int, default=None, help="冒烟测试：只用前 N 棵训练树")
    ap.add_argument("--reward-set", type=str, default="full", help="回报集（graph_cache/rtg_<set>）")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(CFG.train.seed)
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    split = load_val_split()
    train_names = split["train"][:args.max_trees] if args.max_trees else split["train"]
    val_names = split["val"][:max(1, args.max_trees // 3)] if args.max_trees else split["val"]
    t0 = time.time()
    sampler = TransitionSampler(load_cache(train_names, reward_set=args.reward_set),
                                CFG.train.seed, args.expert_weight)
    val_sampler = TransitionSampler(load_cache(val_names, ("expert",), reward_set=args.reward_set), 0, 0.0)
    print(f"[{args.algo}] 设备 {device}；训练树 {len(train_names)}（{len(sampler.trajs)} 条轨迹），"
          f"验证树 {len(val_names)}；读缓存 {(time.time() - t0) / 60:.1f} min", flush=True)

    encoder, agent = build_models(device, args.algo, args.bcq_threshold)
    if args.init_from:
        path = args.init_from if os.path.isabs(args.init_from) \
            else os.path.join(CFG.data.work_root, args.init_from)
        encoder.load_state_dict(torch.load(path, map_location=device, weights_only=False)["encoder"])
        print("编码器初始化自", path)
    enc_t, agent_t = copy.deepcopy(encoder).eval(), copy.deepcopy(agent).eval()
    for p in list(enc_t.parameters()) + list(agent_t.parameters()):
        p.requires_grad_(False)

    params = list(encoder.parameters()) + list(agent.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=CFG.train.weight_decay)
    warmup = max(int(args.steps * CFG.train.warmup_ratio), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        p = min((step - warmup) / max(args.steps - warmup, 1), 1.0)
        return CFG.train.min_lr_ratio + (1 - CFG.train.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    hp = {k: getattr(args, k) for k in ("gamma", "reward_scale", "cql_alpha", "iql_tau", "iql_beta")}
    out_dir = os.path.join(CFG.data.work_root, args.out or f"checkpoints_{args.algo}")
    os.makedirs(out_dir, exist_ok=True)
    meta = {"kind": args.algo, "state_dim": CFG.model.state_dim, "hparams": vars(args),
            "train_trees": len(train_names), "val_trees": val_names}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    log, acc, t0 = [], {}, time.time()
    for step in range(1, args.steps + 1):
        loss, stats = train_step(encoder, agent, enc_t, agent_t,
                                      sampler.sample(args.batch_size, device), hp)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, CFG.train.grad_clip)
        opt.step(); sched.step()
        polyak(enc_t, encoder, args.tau); polyak(agent_t, agent, args.tau)
        for k, v in stats.items():
            acc[k] = acc.get(k, 0.0) + v
        if step % args.log_every == 0 or step == args.steps:
            rec = {"step": step, **{k: v / args.log_every for k, v in acc.items()},
                   "lr": sched.get_last_lr()[0], "minutes": (time.time() - t0) / 60}
            acc = {}
            if step % args.save_every == 0 or step == args.steps:
                rec["val"] = validate(encoder, agent, val_sampler, 64, device)
                ckpt = {"kind": args.algo, "encoder": encoder.state_dict(),
                        "model": agent.state_dict(), "step": step, "meta": meta}
                torch.save(ckpt, os.path.join(out_dir, f"{args.algo}_last.pt"))
                torch.save(ckpt, os.path.join(out_dir, f"{args.algo}_step{step}.pt"))
            log.append(rec)
            print(f"[{args.algo}] step {step}/{args.steps}  " +
                  "  ".join(f"{k} {v:.3f}" for k, v in rec.items()
                            if k not in ("step", "minutes", "val", "lr")) +
                  (f" | val acc1 {rec['val']['acc1_vs_expert']:.3f}" if "val" in rec else "") +
                  f" | {rec['minutes']:.1f} min", flush=True)
            with open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8") as f:
                json.dump(log, f, indent=1)
    print(f"[{args.algo}] 完成，checkpoint: {os.path.join(out_dir, args.algo + '_last.pt')}")


if __name__ == "__main__":
    main()
