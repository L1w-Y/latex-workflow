"""仅微调 Decision Transformer 的 STOP 输出头。

冻结编码器、Transformer 和分枝打分部分；使用平衡的终止/继续窗口训练
STOP 与最佳合法分枝之间的 margin。继续样本包含靠近轨迹末端的困难负样本，
并可用 continue_weight 提高“过早停止”的代价。

用法：
    python train_stop.py --continue-weight 2 --out checkpoints_dt_stopcal/w2
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
from graph_cache import load_cache, load_val_split
from train import WindowSampler, build_models, forward_batch


class StopSampler:
    """平衡采样 STOP、临近终点继续、随机继续三类目标位置。"""

    def __init__(self, trajs: list[dict], context_len: int, rtg_scale: float,
                 seed: int, expert_weight: float = 0.5, hard_width: int = 12):
        self.trajs = [t for t in trajs if len(t["labels"]) >= 2]
        self.expert_idx = [i for i, t in enumerate(self.trajs) if t["policy"] == "expert"]
        self.other_idx = [i for i, t in enumerate(self.trajs) if t["policy"] != "expert"]
        self.expert_weight = expert_weight
        self.hard_width = hard_width
        self.rng = np.random.default_rng(seed)
        self.windows = WindowSampler(
            self.trajs, context_len, rtg_scale, seed=seed, expert_weight=0.0
        )

    def _pick_trajs(self, n: int) -> list[int]:
        if not self.expert_idx or not self.other_idx:
            return self.rng.choice(len(self.trajs), size=n, replace=True).tolist()
        n_exp = int(round(n * self.expert_weight))
        picks = np.concatenate([
            self.rng.choice(self.expert_idx, size=n_exp, replace=True),
            self.rng.choice(self.other_idx, size=n - n_exp, replace=True),
        ])
        self.rng.shuffle(picks)
        return picks.tolist()

    def jobs(self, batch_size: int) -> tuple[list[int], list[int]]:
        """50% STOP、25% 末端困难继续、25% 随机继续。"""
        n_stop = batch_size // 2
        n_hard = (batch_size - n_stop) // 2
        kinds = ["stop"] * n_stop + ["hard"] * n_hard
        kinds += ["continue"] * (batch_size - len(kinds))
        self.rng.shuffle(kinds)
        picks = self._pick_trajs(batch_size)
        times = []
        for i, kind in zip(picks, kinds):
            T = len(self.trajs[i]["labels"])
            if kind == "stop":
                t = T - 1
            elif kind == "hard":
                lo = max(0, T - 1 - self.hard_width)
                t = int(self.rng.integers(lo, T - 1))
            else:
                t = int(self.rng.integers(0, T - 1))
            times.append(t)
        return picks, times

    def build(self, picks: list[int], times: list[int], device):
        K = self.windows.K
        starts = [max(0, t - K + 1) for t in times]
        positions = [t - s for t, s in zip(times, starts)]
        batch = self.windows.sample(
            len(picks), device, picks=picks, starts=starts
        )
        return batch, torch.tensor(positions, dtype=torch.long, device=device)

    def sample(self, batch_size: int, device):
        picks, times = self.jobs(batch_size)
        return self.build(picks, times, device)


def stop_loss(encoder, model, packed, continue_weight: float,
              anchors: list[torch.Tensor], anchor_weight: float):
    batch, positions = packed
    labels = batch[1]
    B, K = labels.shape
    logits, mapped, _ = forward_batch(encoder, model, batch)
    rows = torch.arange(B, device=logits.device) * K + positions
    selected = logits[rows]
    target_labels = mapped[rows]
    n_nodes = selected.size(1) - 1
    target = (target_labels == n_nodes).float()
    branch_best = selected[:, :n_nodes].max(dim=-1).values
    margin = selected[:, n_nodes] - branch_best
    weights = torch.where(
        target > 0.5,
        torch.ones_like(target),
        torch.full_like(target, continue_weight),
    )
    bce = F.binary_cross_entropy_with_logits(
        margin, target, weight=weights
    )
    anchor = sum(
        (p - p0).pow(2).mean()
        for p, p0 in zip(model.head.stop.parameters(), anchors)
    )
    loss = bce + anchor_weight * anchor
    pred = margin > 0
    return loss, {
        "bce": float(bce.item()),
        "anchor": float(anchor.item()),
        "stop_recall": float(pred[target > 0.5].float().mean().item()),
        "false_stop": float(pred[target < 0.5].float().mean().item()),
        "margin_stop": float(margin[target > 0.5].mean().item()),
        "margin_continue": float(margin[target < 0.5].mean().item()),
    }


@torch.no_grad()
def validate(encoder, model, sampler: StopSampler, jobs, batch_size,
             continue_weight, anchors, anchor_weight, device):
    totals: dict[str, float] = {}
    n_batches = 0
    for start in range(0, len(jobs[0]), batch_size):
        picks = jobs[0][start:start + batch_size]
        times = jobs[1][start:start + batch_size]
        _, stats = stop_loss(
            encoder, model, sampler.build(picks, times, device),
            continue_weight, anchors, anchor_weight,
        )
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_dt/dt_best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--continue-weight", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batches-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anchor-weight", type=float, default=1e-4)
    ap.add_argument("--hard-width", type=int, default=12)
    ap.add_argument("--expert-weight", type=float, default=0.5)
    ap.add_argument("--save-every", type=int, default=2)
    ap.add_argument("--val-samples", type=int, default=512)
    ap.add_argument("--reward-set", default="full")
    ap.add_argument("--max-trees", type=int, default=None)
    args = ap.parse_args()

    if args.continue_weight <= 0:
        raise SystemExit("--continue-weight 必须大于 0")
    torch.manual_seed(CFG.train.seed)
    np.random.seed(CFG.train.seed)
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")

    path = args.checkpoint if os.path.isabs(args.checkpoint) \
        else os.path.join(CFG.data.work_root, args.checkpoint)
    base = torch.load(path, map_location=device, weights_only=False)
    if base.get("kind") != "dt":
        raise SystemExit("--checkpoint 必须是当前分枝打分版 DT")
    meta = copy.deepcopy(base["meta"])
    K, rtg_scale = int(meta["context_len"]), float(meta["rtg_scale"])
    encoder, model = build_models(device, meta.get("max_timestep", 1024))
    encoder.load_state_dict(base["encoder"])
    model.load_state_dict(base["model"])

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.stop.parameters():
        parameter.requires_grad_(True)
    encoder.eval()
    model.eval()
    model.head.stop.train()
    anchors = [p.detach().clone() for p in model.head.stop.parameters()]

    split = load_val_split()
    train_names = split["train"][:args.max_trees] if args.max_trees else split["train"]
    val_names = split["val"][:max(1, args.max_trees // 3)] if args.max_trees else split["val"]
    train_sampler = StopSampler(
        load_cache(train_names, reward_set=args.reward_set),
        K, rtg_scale, CFG.train.seed, args.expert_weight, args.hard_width,
    )
    val_sampler = StopSampler(
        load_cache(val_names, reward_set=args.reward_set),
        K, rtg_scale, 2026, args.expert_weight, args.hard_width,
    )
    val_jobs = val_sampler.jobs(args.val_samples)

    params = list(model.head.stop.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    out_dir = args.out if os.path.isabs(args.out) \
        else os.path.join(CFG.data.work_root, args.out)
    os.makedirs(out_dir, exist_ok=True)
    calibration = {
        "base_checkpoint": args.checkpoint,
        "continue_weight": args.continue_weight,
        "epochs": args.epochs,
        "batches_per_epoch": args.batches_per_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "anchor_weight": args.anchor_weight,
        "hard_width": args.hard_width,
        "expert_weight": args.expert_weight,
        "reward_set": args.reward_set,
        "train_trees": len(train_names),
        "val_trees": len(val_names),
    }
    with open(os.path.join(out_dir, "meta_stop.json"), "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=1)

    log = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        totals: dict[str, float] = {}
        for _ in range(args.batches_per_epoch):
            optimizer.zero_grad()
            loss, stats = stop_loss(
                encoder, model, train_sampler.sample(args.batch_size, device),
                args.continue_weight, anchors, args.anchor_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CFG.train.grad_clip)
            optimizer.step()
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
        train_stats = {
            key: value / args.batches_per_epoch for key, value in totals.items()
        }
        val_stats = validate(
            encoder, model, val_sampler, val_jobs, args.batch_size,
            args.continue_weight, anchors, args.anchor_weight, device,
        )
        rec = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "minutes": (time.time() - started) / 60,
        }
        log.append(rec)
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train false_stop={train_stats['false_stop']:.3f} "
            f"stop_recall={train_stats['stop_recall']:.3f} | "
            f"val false_stop={val_stats['false_stop']:.3f} "
            f"stop_recall={val_stats['stop_recall']:.3f}",
            flush=True,
        )
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_meta = copy.deepcopy(meta)
            ckpt_meta["stop_calibration"] = {**calibration, "epoch": epoch}
            checkpoint = {
                "kind": "dt",
                "encoder": encoder.state_dict(),
                "model": model.state_dict(),
                "epoch": epoch,
                "meta": ckpt_meta,
                "stop_val": val_stats,
            }
            torch.save(
                checkpoint, os.path.join(out_dir, f"dt_stop_ep{epoch}.pt")
            )
        with open(os.path.join(out_dir, "stop_log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)

    print("STOP 微调完成：", out_dir)


if __name__ == "__main__":
    main()
