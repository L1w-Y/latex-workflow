"""KPI修复后的策略诊断：配对结果、动作粒度与KPI输入反事实。

只读取现有checkpoint、缓存与选模评估结果，不重新训练或在线计算候选KPI。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
sys.path.insert(0, str(ROOT / "dt_pruning"))

from config import CFG  # noqa: E402
from graph_cache_v2 import load_cache, load_split  # noqa: E402
from kpi_scales_v2 import activate_v2_kpi_scales  # noqa: E402
from train_v2 import WindowSamplerV2, build_models_from_meta, forward_batch  # noqa: E402

KPI = ("hierarchy", "crowding", "balance", "light", "potential", "stability")
WEIGHTS = {
    "hierarchy": 5.0, "crowding": 10.0, "balance": 8.0,
    "light": 8.0, "potential": 2.0, "stability": 5.0,
}


def _paired_rollout_diagnosis(result_dir: Path, scales: dict) -> dict:
    rows, traces = {}, {}
    for tag in ("full", "no_kpi"):
        with (result_dir / f"eval_{tag}_curve.csv").open(encoding="utf-8") as f:
            rows[tag] = {r["tree"]: r for r in csv.DictReader(f)}
        with (result_dir / f"actions_{tag}_curve.json").open(encoding="utf-8") as f:
            traces[tag] = json.load(f)
    trees = sorted(set(rows["full"]) & set(rows["no_kpi"]))

    def score_delta(row: dict) -> float:
        return float(sum(
            WEIGHTS[k] * float(row[f"{k}_delta"]) / scales[k] for k in KPI))

    def auc(row: dict, trace: list[dict]) -> float:
        x = [0.0] + [float(s["pruning_ratio_after"]) for s in trace]
        y = [0.0] + [sum(
            WEIGHTS[k] * (
                float(s["kpi_after"][k]) - float(row[f"{k}_before"])
            ) / scales[k] for k in KPI
        ) for s in trace]
        return float(np.trapz(y, x) / 0.32)

    per_tree = []
    for tree in trees:
        rf, rn = rows["full"][tree], rows["no_kpi"][tree]
        tf, tn = traces["full"][tree], traces["no_kpi"][tree]
        per_tree.append({
            "tree": tree,
            "cap_diff_full_minus_no_kpi": score_delta(rf) - score_delta(rn),
            "auc_diff_full_minus_no_kpi": auc(rf, tf) - auc(rn, tn),
            "steps_full": len(tf),
            "steps_no_kpi": len(tn),
            "mean_ratio_increment_full": (
                float(rf["final_pruning_ratio"]) / max(len(tf), 1)),
            "mean_ratio_increment_no_kpi": (
                float(rn["final_pruning_ratio"]) / max(len(tn), 1)),
            **{
                f"{k}_weighted_diff": WEIGHTS[k] * (
                    float(rf[f"{k}_delta"]) - float(rn[f"{k}_delta"])
                ) / scales[k] for k in KPI
            },
        })

    def paired_summary(key: str) -> dict:
        values = np.asarray([r[key] for r in per_tree], dtype=float)
        try:
            p = float(wilcoxon(values).pvalue)
        except ValueError:
            p = 1.0
        rng = np.random.default_rng(42)
        means = np.asarray([
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(10000)
        ])
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "full_wins": int((values > 0).sum()),
            "no_kpi_wins": int((values < 0).sum()),
            "wilcoxon_p": p,
            "bootstrap_mean_ci95": [
                float(np.quantile(means, 0.025)),
                float(np.quantile(means, 0.975)),
            ],
        }

    return {
        "n_trees": len(trees),
        "cap_pairing": paired_summary("cap_diff_full_minus_no_kpi"),
        "auc_pairing": paired_summary("auc_diff_full_minus_no_kpi"),
        "mean_steps": {
            tag: float(np.mean([r[f"steps_{tag}"] for r in per_tree]))
            for tag in ("full", "no_kpi")
        },
        "mean_ratio_increment_per_action": {
            tag: float(np.mean([
                r[f"mean_ratio_increment_{tag}"] for r in per_tree
            ])) for tag in ("full", "no_kpi")
        },
        "mean_weighted_dimension_diff": {
            k: float(np.mean([r[f"{k}_weighted_diff"] for r in per_tree]))
            for k in KPI
        },
        "per_tree": per_tree,
    }


def _branch_counterfactual(checkpoint: Path, batch_size: int) -> dict:
    device = torch.device(CFG.train.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    scales = activate_v2_kpi_scales()
    target_scales = np.asarray([scales[k] for k in KPI], dtype=np.float32)
    encoder, model = build_models_from_meta(device, meta)
    encoder.load_state_dict(ckpt["encoder"])
    model.load_state_dict(ckpt["model"])
    encoder.eval()
    model.eval()

    val = load_cache(
        load_split()["val"], ("gardening",), meta.get("reward_set", "full"))
    sampler = WindowSamplerV2(
        val, meta["context_len"], meta["rtg_scale"], 0, 1.0,
        dk_scales=target_scales,
        kpi_efficiency_ref=meta.get("kpi_efficiency_ref"))
    jobs = [(i, s) for i, tr in enumerate(sampler.trajs)
            for s in range(0, max(len(tr["labels"]) - 1, 1), sampler.K)]

    changed = valid_states = correct_full = correct_zero = 0
    abs_logit_shift, candidate_rho = [], []
    with torch.no_grad():
        for pos in range(0, len(jobs), batch_size):
            chunk = jobs[pos:pos + batch_size]
            batch = sampler.sample(
                len(chunk), device,
                [x[0] for x in chunk], [x[1] for x in chunk])
            model.head.use_kpi_branch = True
            full = forward_batch(encoder, model, batch)
            model.head.use_kpi_branch = False
            zero = forward_batch(encoder, model, batch)
            labels = full[4]
            mask = labels >= 0
            if not mask.any():
                continue
            full_logits = full[0][mask]
            zero_logits = zero[0][mask]
            y = labels[mask]
            pred_full = full_logits.argmax(-1)
            pred_zero = zero_logits.argmax(-1)
            changed += int((pred_full != pred_zero).sum().item())
            valid_states += int(mask.sum().item())
            correct_full += int((pred_full == y).sum().item())
            correct_zero += int((pred_zero == y).sum().item())
            finite = torch.isfinite(full_logits) & torch.isfinite(zero_logits)
            abs_logit_shift.extend(
                (full_logits - zero_logits).abs()[finite].cpu().tolist())
            for i in range(len(full_logits)):
                legal = finite[i]
                if int(legal.sum()) < 3:
                    continue
                rho = spearmanr(
                    full_logits[i, legal].cpu().numpy(),
                    zero_logits[i, legal].cpu().numpy()).statistic
                if np.isfinite(rho):
                    candidate_rho.append(float(rho))
    model.head.use_kpi_branch = True
    return {
        "n_selected_action_states": valid_states,
        "argmax_changed_fraction_when_kpi_zeroed": (
            changed / max(valid_states, 1)),
        "selected_action_top1_full": correct_full / max(valid_states, 1),
        "selected_action_top1_kpi_zeroed": correct_zero / max(valid_states, 1),
        "mean_abs_candidate_logit_shift": float(np.mean(abs_logit_shift)),
        "mean_candidate_rank_correlation_full_vs_zeroed": float(
            np.mean(candidate_rho)),
        "note": (
            "该反事实只判断Branch是否使用KPI输入；不包含所有候选动作的真实ΔK，"
            "不能替代候选级排序真值诊断。"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="experiments/checkpoints_dt_kpifix_v2/dt_best.pt")
    ap.add_argument(
        "--result-dir",
        default="experiments/results/dev/kpi_redesign_gate_v2/model_val")
    ap.add_argument(
        "--out",
        default="experiments/results/dev/kpi_redesign_v2/policy_diagnosis.json")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    scales = activate_v2_kpi_scales()
    report = {
        "paired_rollout": _paired_rollout_diagnosis(
            ROOT / args.result_dir, scales),
        "branch_kpi_counterfactual": _branch_counterfactual(
            ROOT / args.checkpoint, args.batch_size),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
