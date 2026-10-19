"""训练 schema=2 多任务 DT。

损失：L_branch + λ_stop L_stop + λ_kpi L_ΔK。只读取 graph_cache_v2，
checkpoint 目录名必须含 ``_v2``，避免覆盖 v1。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from ablation_v2 import mask_graph_features
from config import CFG
from dt_model import PAD_LABEL, map_labels
from dt_model_v2 import DecisionTransformerV2
from encoder import (BRANCH_FEAT_DIM_V2, GLOBAL_FEAT_DIM_V2,
                     BranchGraphEncoder)
from graph_cache_v2 import cached_to_data, load_cache, load_split
from kpi_scales_v2 import activate_v2_kpi_scales
from reward import KPI_KEYS
from v2_manifest import load_manifest, manifest_hash

KPI_EFFICIENCY_REF = 0.01


def attach_candidate_targets(trajs: list[dict], root: str | os.PathLike,
                             required: bool = True) -> int:
    """按(tree, policy, variant)挂接并列候选监督，不修改磁盘图缓存。"""
    root = Path(root)
    by_tree: dict[str, list[dict]] = {}
    for tree in sorted({t["tree"] for t in trajs}):
        path = root / f"{tree}.pkl"
        if not path.exists():
            if required:
                raise FileNotFoundError(f"缺少候选KPI缓存：{path}")
            continue
        with path.open("rb") as f:
            by_tree[tree] = pickle.load(f)
    attached = 0
    for tr in trajs:
        candidates = {
            (x["policy"], int(x.get("variant", 0))): x
            for x in by_tree.get(tr["tree"], [])
        }
        item = candidates.get((tr["policy"], int(tr.get("variant", 0))))
        if item is None:
            if required:
                raise ValueError(
                    f"{tr['tree']}/{tr['policy']}/{tr.get('variant', 0)}"
                    " 缺候选监督")
            continue
        if len(item["steps"]) != len(tr["labels"]):
            raise ValueError(f"{tr['tree']} 候选监督步数与图缓存不一致")
        tr["candidate_steps"] = item["steps"]
        attached += 1
    return attached


class WindowSamplerV2:
    def __init__(self, trajs: list[dict], context_len: int, rtg_scale: float,
                 seed: int = 42, knowledge_weight: float = 0.5,
                 use_rule_features: bool = True,
                 use_kpi_features: bool = True,
                 dk_scales: np.ndarray | None = None,
                 kpi_efficiency_ref: float | None = None):
        self.trajs = [t for t in trajs if len(t["labels"])]
        self.knowledge_idx = [
            i for i, t in enumerate(self.trajs) if t["policy"] in ("gardening", "topk")]
        self.other_idx = [i for i in range(len(self.trajs)) if i not in self.knowledge_idx]
        self.knowledge_weight = knowledge_weight if self.knowledge_idx else 0.0
        self.K, self.rtg_scale = context_len, rtg_scale
        self.use_rule_features = use_rule_features
        self.use_kpi_features = use_kpi_features
        self.dk_scales = (
            np.asarray(dk_scales, dtype=np.float32) if dk_scales is not None
            else np.ones(6, dtype=np.float32))
        self.kpi_efficiency_ref = kpi_efficiency_ref
        self.candidate_cap = max((
            len(step["indices"])
            for tr in self.trajs for step in tr.get("candidate_steps", [])
        ), default=0)
        if self.candidate_cap and self.kpi_efficiency_ref is None:
            raise ValueError("候选KPI监督必须指定kpi_efficiency_ref")
        self.rng = np.random.default_rng(seed)

    def _pick(self, size: int):
        if self.knowledge_weight <= 0 or not self.other_idx:
            return self.rng.choice(len(self.trajs), size=size, replace=True)
        n = int(round(size * self.knowledge_weight))
        return np.concatenate([
            self.rng.choice(self.knowledge_idx, size=n, replace=True),
            self.rng.choice(self.other_idx, size=size - n, replace=True),
        ])

    def sample(self, size: int, device, picks=None, starts=None):
        picks = self._pick(size) if picks is None else np.asarray(picks)
        graphs, labels, rtg, ts, mask, dk, dk_mask, rewards = [], [], [], [], [], [], [], []
        candidate_idx, candidate_dk, candidate_mask = [], [], []
        for j, idx in enumerate(picks):
            tr = self.trajs[int(idx)]
            total = len(tr["labels"])
            start = int(starts[j]) if starts is not None else (
                int(self.rng.integers(0, total - self.K + 1)) if total > self.K else 0)
            end = min(start + self.K, total)
            length, pad = end - start, self.K - (end - start)
            graphs.extend(tr["graphs"][start:end] + [tr["graphs"][end - 1]] * pad)
            labels.append(list(tr["labels"][start:end]) + [PAD_LABEL] * pad)
            rtg.append(list(tr["rtg"][start:end] / self.rtg_scale) + [0.0] * pad)
            ts.append(list(range(start, end)) + [0] * pad)
            mask.append([1.0] * length + [0.0] * pad)
            dk_target = tr["dk"][start:end] / self.dk_scales
            if self.kpi_efficiency_ref is not None:
                ratio = np.asarray([
                    max(float(tr["graphs"][t + 1]["g"][0])
                        - float(tr["graphs"][t]["g"][0]), 1e-8)
                    if tr["labels"][t] >= 0 and t + 1 < total else 1.0
                    for t in range(start, end)
                ], dtype=np.float32)
                dk_target = dk_target * (
                    self.kpi_efficiency_ref / ratio[:, None])
            dk.append(dk_target.tolist() + [[0.0] * 6] * pad)
            dk_mask.append(tr["dk_mask"][start:end].tolist() + [False] * pad)
            rewards.append(list(tr["rewards"][start:end]) + [0.0] * pad)
            if self.candidate_cap:
                for t in range(start, end):
                    step = tr["candidate_steps"][t]
                    n = len(step["indices"])
                    c_idx = np.full(self.candidate_cap, -1, dtype=np.int64)
                    c_dk = np.zeros((self.candidate_cap, 6), dtype=np.float32)
                    c_mask = np.zeros(self.candidate_cap, dtype=bool)
                    if n:
                        c_idx[:n] = step["indices"]
                        c_dk[:n] = (
                            step["dk"] / self.dk_scales
                            * (self.kpi_efficiency_ref
                               / np.maximum(
                                   step["ratio_delta"][:, None], 1e-8)))
                        c_mask[:n] = True
                    candidate_idx.append(c_idx)
                    candidate_dk.append(c_dk)
                    candidate_mask.append(c_mask)
                for _ in range(pad):
                    candidate_idx.append(
                        np.full(self.candidate_cap, -1, dtype=np.int64))
                    candidate_dk.append(
                        np.zeros((self.candidate_cap, 6), dtype=np.float32))
                    candidate_mask.append(
                        np.zeros(self.candidate_cap, dtype=bool))
        batch = (
            mask_graph_features(
                Batch.from_data_list([cached_to_data(g) for g in graphs]).to(device),
                use_rule_features=self.use_rule_features,
                use_kpi_features=self.use_kpi_features),
            torch.tensor(labels, dtype=torch.long, device=device),
            torch.tensor(rtg, dtype=torch.float32, device=device),
            torch.tensor(ts, dtype=torch.long, device=device),
            torch.tensor(mask, dtype=torch.float32, device=device),
            torch.tensor(dk, dtype=torch.float32, device=device),
            torch.tensor(dk_mask, dtype=torch.bool, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
        )
        if self.candidate_cap:
            batch += (
                torch.tensor(
                    np.asarray(candidate_idx).reshape(size, self.K, -1),
                    dtype=torch.long, device=device),
                torch.tensor(
                    np.asarray(candidate_dk).reshape(size, self.K, -1, 6),
                    dtype=torch.float32, device=device),
                torch.tensor(
                    np.asarray(candidate_mask).reshape(size, self.K, -1),
                    dtype=torch.bool, device=device),
            )
        return batch


def build_models(device, max_timestep: int = 1024,
                 explicit_kpi_branch: bool = False,
                 monotonic_kpi_branch: bool = False,
                 use_kpi_branch: bool = True):
    m = CFG.model
    encoder = BranchGraphEncoder(
        in_dim=BRANCH_FEAT_DIM_V2, global_dim=GLOBAL_FEAT_DIM_V2,
        hidden=m.gnn_hidden, heads=m.gnn_heads, layers=m.gnn_layers,
        state_dim=m.state_dim, dropout=m.dropout).to(device)
    model = DecisionTransformerV2(
        state_dim=m.state_dim, node_dim=encoder.node_dim,
        hidden_dim=m.hidden_dim, n_layer=m.n_layer, n_head=m.n_head,
        dropout=m.dropout, max_timestep=max_timestep,
        explicit_kpi_branch=explicit_kpi_branch,
        monotonic_kpi_branch=monotonic_kpi_branch,
        use_kpi_branch=use_kpi_branch,
        kpi_weights=tuple(CFG.reward.kpi_weights[k] for k in KPI_KEYS)).to(device)
    return encoder, model


def build_models_from_meta(device, meta: dict):
    revision = meta.get("model_revision", "legacy-v2")
    return build_models(
        device, meta.get("max_timestep", 1024),
        revision == "kpi-explicit-v3",
        revision == "kpi-ranking-v4",
        meta.get("use_kpi_branch", meta.get("use_kpi_aux", True)))


def forward_batch(encoder, model, batch, return_candidate: bool = False):
    graph, labels, rtg, ts, mask, dk, dk_mask, rewards = batch[:8]
    batch_size, context_len = labels.shape
    state, nodes, legal = encoder(graph)
    mapped = map_labels(labels.reshape(-1), nodes.size(1))
    action_features = model.action_features(nodes, mapped)
    ctx = model(
        state.view(batch_size, context_len, -1),
        action_features.view(batch_size, context_len, -1),
        rtg.unsqueeze(-1), ts, mask)
    branch, stop, dk_all, net_all = model.head(ctx.reshape(-1, ctx.size(-1)), nodes, legal)
    gather_idx = labels.reshape(-1).clamp(0, nodes.size(1) - 1)
    dk_pred = dk_all[torch.arange(len(gather_idx), device=nodes.device), gather_idx]
    net_pred = net_all[torch.arange(len(gather_idx), device=nodes.device), gather_idx]
    outputs = (
        branch, stop, dk_pred, net_pred, labels.reshape(-1), dk.reshape(-1, 6),
        dk_mask.reshape(-1), rewards.reshape(-1))
    if not return_candidate:
        return outputs
    if len(batch) < 11:
        raise ValueError("启用候选KPI损失但batch中没有候选监督")
    candidate_idx, candidate_dk, candidate_mask = batch[8:11]
    flat_idx = candidate_idx.reshape(-1, candidate_idx.size(-1))
    flat_target = candidate_dk.reshape(-1, candidate_dk.size(-2), 6)
    flat_mask = candidate_mask.reshape(-1, candidate_mask.size(-1))
    safe_idx = flat_idx.clamp(0, dk_all.size(1) - 1)
    rows = torch.arange(len(safe_idx), device=dk_all.device).unsqueeze(1)
    candidate_pred = dk_all[rows, safe_idx]
    flat_mask = flat_mask & (flat_idx >= 0)
    return outputs, (candidate_pred, flat_target, flat_mask)


def candidate_losses(candidate, kpi_weights: torch.Tensor):
    pred, target, mask = candidate
    if not mask.any():
        zero = pred.sum() * 0.0
        return zero, zero, 0.0
    regression = F.smooth_l1_loss(pred[mask], target[mask])
    weights = kpi_weights.to(pred)
    pred_utility = (pred * weights).sum(-1).masked_fill(~mask, -1e9)
    true_utility = (target * weights).sum(-1).masked_fill(~mask, -1e9)
    valid_rows = mask.sum(-1) >= 2
    if valid_rows.any():
        best = true_utility[valid_rows].argmax(-1)
        ranking = F.cross_entropy(pred_utility[valid_rows], best)
        rank_acc = float(
            (pred_utility[valid_rows].argmax(-1) == best).float().mean().item())
    else:
        ranking = regression * 0.0
        rank_acc = 0.0
    return regression, ranking, rank_acc


def losses(outputs, lambda_stop: float, lambda_kpi: float,
           stop_pos_weight: float = 8.0, candidate=None,
           kpi_weights: torch.Tensor | None = None):
    branch, stop, dk_pred, net_pred, labels, dk, dk_mask, rewards = outputs
    valid = labels != PAD_LABEL
    stop_target = (labels == -1).float()
    stop_loss = F.binary_cross_entropy_with_logits(
        stop[valid], stop_target[valid],
        pos_weight=stop.new_tensor(stop_pos_weight))
    branch_mask = valid & (labels >= 0)
    branch_loss = (F.cross_entropy(branch[branch_mask], labels[branch_mask])
                   if branch_mask.any() else stop.sum() * 0.0)
    aux_mask = valid & dk_mask & (labels >= 0)
    kpi_loss = (F.smooth_l1_loss(dk_pred[aux_mask], dk[aux_mask])
                if aux_mask.any() else stop.sum() * 0.0)
    candidate_reg = stop.sum() * 0.0
    candidate_rank = stop.sum() * 0.0
    candidate_rank_acc = 0.0
    if candidate is not None:
        if kpi_weights is None:
            raise ValueError("候选KPI损失需要六维权重")
        candidate_reg, candidate_rank, candidate_rank_acc = candidate_losses(
            candidate, kpi_weights)
        # 候选监督已包含实际动作；不再重复使用只含实际动作的KPI回归损失。
        kpi_loss = candidate_reg + candidate_rank
    net_loss = (F.smooth_l1_loss(net_pred[aux_mask], rewards[aux_mask] / 1000.0)
                if aux_mask.any() else stop.sum() * 0.0)
    total = branch_loss + lambda_stop * stop_loss + lambda_kpi * (kpi_loss + 0.1 * net_loss)
    with torch.no_grad():
        branch_acc = ((branch[branch_mask].argmax(-1) == labels[branch_mask]).float().mean().item()
                      if branch_mask.any() else 0.0)
        stop_acc = ((stop[valid] >= 0) == (labels[valid] == -1)).float().mean().item()
    return total, {
        "branch_loss": branch_loss.item(), "stop_loss": stop_loss.item(),
        "kpi_loss": kpi_loss.item(), "net_loss": net_loss.item(),
        "candidate_reg": candidate_reg.item(),
        "candidate_rank": candidate_rank.item(),
        "candidate_rank_acc": candidate_rank_acc,
        "branch_acc": branch_acc, "stop_acc": stop_acc,
    }


@torch.no_grad()
def validate(encoder, model, sampler, batch_size, device,
             lambda_stop, lambda_kpi, stop_pos_weight):
    encoder.eval(); model.eval()
    jobs = [(i, s) for i, tr in enumerate(sampler.trajs)
            for s in range(0, max(len(tr["labels"]) - 1, 1), sampler.K)]
    sums = {k: 0.0 for k in (
        "loss", "branch_loss", "stop_loss", "kpi_loss", "net_loss",
        "candidate_reg", "candidate_rank", "candidate_rank_acc",
        "branch_acc", "stop_acc")}
    batches = 0
    for pos in range(0, len(jobs), batch_size):
        chunk = jobs[pos:pos + batch_size]
        batch = sampler.sample(
            len(chunk), device, [x[0] for x in chunk], [x[1] for x in chunk])
        forwarded = forward_batch(
            encoder, model, batch, return_candidate=bool(sampler.candidate_cap))
        out, candidate = (
            forwarded if sampler.candidate_cap else (forwarded, None))
        loss, metrics = losses(
            out, lambda_stop, lambda_kpi, stop_pos_weight, candidate,
            model.head.kpi_weights)
        sums["loss"] += loss.item()
        for key, value in metrics.items():
            sums[key] += value
        batches += 1
    return {key: value / max(batches, 1) for key, value in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--context-len", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batches-per-epoch", type=int, default=200)
    ap.add_argument("--lr", type=float, default=CFG.train.lr)
    ap.add_argument("--rtg-scale", type=float, default=1000.0)
    ap.add_argument("--lambda-stop", type=float, default=1.0)
    ap.add_argument("--lambda-kpi", type=float, default=0.2)
    ap.add_argument("--stop-pos-weight", type=float, default=8.0,
                    help="STOP 正样本权重，缓解每条轨迹仅一个 STOP 的类别不平衡")
    ap.add_argument("--knowledge-weight", type=float, default=0.5)
    ap.add_argument("--policies", default="gardening,topk,kpi")
    ap.add_argument("--reward-set", choices=("full", "norq", "nori", "norc"),
                    default="full")
    ap.add_argument("--cache-root", default=None,
                    help="可选并列v2缓存目录；相对 experiments/")
    ap.add_argument("--no-rule-features", action="store_true")
    ap.add_argument("--no-kpi-features", action="store_true")
    ap.add_argument("--no-kpi-aux", action="store_true")
    ap.add_argument("--explicit-kpi-branch", action="store_true")
    ap.add_argument("--monotonic-kpi-branch", action="store_true")
    ap.add_argument("--candidate-kpi-root", default=None,
                    help="并列候选KPI监督目录；相对experiments/")
    ap.add_argument("--kpi-efficiency-ref", type=float,
                    default=KPI_EFFICIENCY_REF,
                    help="把ΔK换算到每1%%剪枝量的固定参考尺度")
    ap.add_argument("--ablation", default="full")
    ap.add_argument("--init-from", default=None,
                    help="可选：用同一 manifest 的 BC v2 checkpoint 初始化图编码器")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--max-trees", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", default="checkpoints_dt_v2")
    args = ap.parse_args()
    if "_v2" not in os.path.basename(args.out):
        raise SystemExit("v2 checkpoint 目录名必须含 _v2，避免覆盖 v1")
    if args.max_batches is not None:
        args.batches_per_epoch = min(args.batches_per_epoch, args.max_batches)
    if args.monotonic_kpi_branch and not args.no_kpi_aux \
            and not args.candidate_kpi_root:
        raise SystemExit("单调KPI融合必须提供--candidate-kpi-root候选级真值")
    if args.explicit_kpi_branch and args.monotonic_kpi_branch:
        raise SystemExit("--explicit-kpi-branch与--monotonic-kpi-branch不能同时使用")

    torch.manual_seed(CFG.train.seed)
    np.random.seed(CFG.train.seed)
    kpi_scales = activate_v2_kpi_scales()
    dk_scales = np.asarray([kpi_scales[k] for k in KPI_KEYS], dtype=np.float32)
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    policies = tuple(x.strip() for x in args.policies.split(",") if x.strip())
    allowed = {"gardening", "topk", "kpi", "random"}
    if not policies or not set(policies) <= allowed:
        raise SystemExit(f"--policies 只能来自 {sorted(allowed)}")
    cache_root = args.cache_root
    if cache_root and not os.path.isabs(cache_root):
        cache_root = os.path.join(CFG.data.work_root, cache_root)

    split = load_split()
    train_names = split["train"][:args.max_trees] if args.max_trees else split["train"]
    n_val = max(1, (args.max_trees or 0) // 2) if args.max_trees else len(split["val"])
    val_names = split["val"][:n_val]
    train_trajs = load_cache(
        train_names, policies, args.reward_set, cache_root)
    val_policies = policies if args.ablation == "no_gardening" else ("gardening",)
    val_trajs = load_cache(
        val_names, val_policies, args.reward_set, cache_root)
    if not train_trajs or not val_trajs:
        raise SystemExit("v2 训练/验证缓存为空：先运行 graph_cache_v2.py")
    candidate_root = args.candidate_kpi_root
    if candidate_root and not os.path.isabs(candidate_root):
        candidate_root = os.path.join(CFG.data.work_root, candidate_root)
    if candidate_root and not args.no_kpi_aux:
        n_train = attach_candidate_targets(train_trajs, candidate_root)
        n_val = attach_candidate_targets(val_trajs, candidate_root)
        print(f"候选KPI监督：train={n_train}条，val={n_val}条")
    sampler = WindowSamplerV2(
        train_trajs, args.context_len, args.rtg_scale,
        CFG.train.seed, args.knowledge_weight,
        not args.no_rule_features, not args.no_kpi_features, dk_scales,
        args.kpi_efficiency_ref if candidate_root and not args.no_kpi_aux else None)
    val_sampler = WindowSamplerV2(
        val_trajs, args.context_len, args.rtg_scale, 0, 1.0,
        not args.no_rule_features, not args.no_kpi_features, dk_scales,
        args.kpi_efficiency_ref if candidate_root and not args.no_kpi_aux else None)
    encoder, model = build_models(
        device, explicit_kpi_branch=args.explicit_kpi_branch,
        monotonic_kpi_branch=args.monotonic_kpi_branch,
        use_kpi_branch=not args.no_kpi_aux)
    if args.init_from:
        init_path = args.init_from if os.path.isabs(args.init_from) else os.path.join(
            CFG.data.work_root, args.init_from)
        init = torch.load(init_path, map_location=device, weights_only=False)
        if init.get("kind") != "bc_v2":
            raise SystemExit("--init-from 必须是 kind=bc_v2 checkpoint")
        if init.get("meta", {}).get("manifest_hash") != manifest_hash(load_manifest()):
            raise SystemExit("BC checkpoint 与当前 v2 manifest 不一致")
        init_meta = init.get("meta", {})
        if (init_meta.get("use_rule_features", True) != (not args.no_rule_features)
                or init_meta.get("use_kpi_features", True)
                != (not args.no_kpi_features)):
            raise SystemExit("BC checkpoint 的特征开关与当前消融不一致")
        encoder.load_state_dict(init["encoder"])
        print("编码器初始化自", init_path)
    params = list(encoder.parameters()) + list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=CFG.train.weight_decay)
    total_steps = args.epochs * args.batches_per_epoch
    warmup = max(int(total_steps * CFG.train.warmup_ratio), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = min((step - warmup) / max(total_steps - warmup, 1), 1.0)
        return CFG.train.min_lr_ratio + (1 - CFG.train.min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    out_dir = os.path.join(CFG.data.work_root, args.out)
    os.makedirs(out_dir, exist_ok=True)
    manifest = load_manifest()
    meta = {
        "kind": "dt_v2", "schema_version": 2, "dataset_version": CFG.data.v2_version,
        "manifest_hash": manifest_hash(manifest), "context_len": args.context_len,
        "rtg_scale": args.rtg_scale, "max_timestep": model.max_timestep,
        "policies": list(policies), "lambda_stop": args.lambda_stop,
        "lambda_kpi": 0.0 if args.no_kpi_aux else args.lambda_kpi,
        "stop_pos_weight": args.stop_pos_weight,
        "reward_set": args.reward_set, "cache_root": args.cache_root,
        "use_rule_features": not args.no_rule_features,
        "use_kpi_features": not args.no_kpi_features,
        "use_kpi_aux": not args.no_kpi_aux, "ablation": args.ablation,
        "use_kpi_branch": not args.no_kpi_aux,
        "model_revision": (
            "kpi-ranking-v4" if args.monotonic_kpi_branch
            else "kpi-explicit-v3" if args.explicit_kpi_branch else "legacy-v2"),
        "kpi_target": (
            "delta_over_train_q95_per_1pct_pruning"
            if candidate_root and not args.no_kpi_aux
            else "delta_over_train_q95"),
        "candidate_kpi_root": args.candidate_kpi_root,
        "kpi_efficiency_ref": (
            args.kpi_efficiency_ref if candidate_root and not args.no_kpi_aux
            else None),
        "kpi_scales": kpi_scales,
        "init_from": args.init_from,
        "feature_dim": BRANCH_FEAT_DIM_V2,
        "global_dim": GLOBAL_FEAT_DIM_V2, "train_trees": train_names,
        "val_trees": val_names, "checkpoint_selection": "val_multitask_loss",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log, best, started = [], float("inf"), time.time()
    for epoch in range(1, args.epochs + 1):
        encoder.train(); model.train()
        sums = {k: 0.0 for k in (
            "loss", "branch_loss", "stop_loss", "kpi_loss", "net_loss",
            "candidate_reg", "candidate_rank", "candidate_rank_acc",
            "branch_acc", "stop_acc")}
        for _ in range(args.batches_per_epoch):
            batch = sampler.sample(args.batch_size, device)
            forwarded = forward_batch(
                encoder, model, batch,
                return_candidate=bool(sampler.candidate_cap))
            out, candidate = (
                forwarded if sampler.candidate_cap else (forwarded, None))
            loss, metrics = losses(
                out, args.lambda_stop,
                0.0 if args.no_kpi_aux else args.lambda_kpi,
                args.stop_pos_weight, candidate, model.head.kpi_weights)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CFG.train.grad_clip)
            opt.step(); scheduler.step()
            sums["loss"] += loss.item()
            for key, value in metrics.items():
                sums[key] += value
        rec = {k: v / args.batches_per_epoch for k, v in sums.items()}
        rec.update(epoch=epoch, lr=scheduler.get_last_lr()[0],
                   minutes=(time.time() - started) / 60)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            rec["val"] = validate(
                encoder, model, val_sampler, args.batch_size, device,
                args.lambda_stop,
                0.0 if args.no_kpi_aux else args.lambda_kpi,
                args.stop_pos_weight)
            ckpt = {"kind": "dt_v2", "encoder": encoder.state_dict(),
                    "model": model.state_dict(), "epoch": epoch,
                    "meta": meta, "val": rec["val"]}
            torch.save(ckpt, os.path.join(out_dir, "dt_last.pt"))
            if rec["val"]["loss"] < best:
                best = rec["val"]["loss"]
                torch.save(ckpt, os.path.join(out_dir, "dt_best.pt"))
        log.append(rec)
        print(f"epoch {epoch}/{args.epochs} loss={rec['loss']:.4f} "
              f"branch_acc={rec['branch_acc']:.3f} stop_acc={rec['stop_acc']:.3f}"
              + (f" val={rec['val']['loss']:.4f}" if "val" in rec else "")
              + f" | {rec['minutes']:.1f} min", flush=True)
        with open(os.path.join(out_dir, "loss_log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
