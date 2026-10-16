"""
Visualise step-by-step pruning rollouts produced by the trained model.

Reuses the already-verified ranker / policy loader from run_benchmark.py to
avoid duplicating model definitions or feature-construction logic. We import
the helpers (now that run_benchmark.py is guarded with `if __name__ ==
'__main__':`) and call them inside a snapshot-recording rollout loop.

Output: an X-Z scatter grid (1 x N_STEPS) for the chosen tree under
../论文/figures/learned_rollout_<TreeName>.png.

Run with the base env (Python 3.12 + torch):

    & D:\\PROJECTS\\deeplearning\\python.exe plot_learned_rollout.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_trajectories import (
    MAX_PRUNING_RATIO,
    STOP_ACTION,
    _branch_to_nodes,
    _build_graph,
    _point_branch_ids,
    _read_point_cloud,
    _read_skeleton,
    _remove_branch_state,
    _removed_branch_ids_for_action,
    _root_branch_id,
)
from utils import calculate_step_reward

from run_benchmark import (
    MAX_BENCHMARK_STEPS,
    _choose_pointcloud_file,
    _load_rl_policy,
    _select_rl_action,
    _state_payload,
    device as RB_DEVICE,
)


DEFAULT_DATA_ROOT = Path("./data")
DEFAULT_CHECKPOINT = Path("./checkpoints/graph_gnn/graph_candidate_ranker_best.pt")
DEFAULT_OUT_DIR = Path("../论文/figures")
N_STEPS_TO_SHOW = 4
GREY_RGBA = (0.35, 0.35, 0.35, 0.45)
RED_PC_RGBA = (0.85, 0.10, 0.10, 0.85)
RED_SK_RGBA = (0.85, 0.05, 0.05, 1.00)
GREY_SIZE = 0.25
RED_PC_SIZE = 4.0
RED_SK_SIZE = 6.0


def rollout_with_snapshots(
    tree_dir: Path,
    rl_model: dict[str, Any],
    max_steps: int = MAX_BENCHMARK_STEPS,
) -> dict[str, Any]:
    point_cloud = _read_point_cloud(_choose_pointcloud_file(tree_dir))
    skeleton = _read_skeleton(tree_dir / "skeleton" / "skeleton.txt")
    graph = _build_graph(skeleton)
    branch_nodes = _branch_to_nodes(graph)
    root_branch = _root_branch_id(graph)
    point_branch_ids = _point_branch_ids(point_cloud, skeleton)
    initial_branch_count = len([b for b in branch_nodes if int(b) >= 0 and int(b) != int(root_branch)])
    max_allowed_cuts = max(1, int(np.floor(MAX_PRUNING_RATIO * initial_branch_count + 1e-8)))

    snapshots: list[dict[str, Any]] = []
    pruned_ids: set[int] = set()
    cumulative = 0.0

    for step in range(max_steps):
        valid = [b for b in branch_nodes if int(b) >= 0 and int(b) != int(root_branch)]
        ratio = len(pruned_ids) / max(1, initial_branch_count)
        if (not valid) or ratio >= MAX_PRUNING_RATIO or len(pruned_ids) >= max_allowed_cuts:
            break

        chosen, _extra = _select_rl_action(
            rl_model,
            point_cloud,
            skeleton,
            step_idx=step,
            max_steps=max_steps,
            pruning_ratio=ratio,
        )
        if int(chosen) == STOP_ACTION:
            break

        removed_set = _removed_branch_ids_for_action(graph, int(chosen), branch_nodes)
        if not removed_set:
            branch_nodes.pop(int(chosen), None)
            continue
        if len(pruned_ids.union(removed_set)) > max_allowed_cuts:
            break

        next_pc, next_sk, next_pbids, _, removed_n = _remove_branch_state(
            point_cloud, skeleton, point_branch_ids, int(chosen), graph, branch_nodes,
        )
        if removed_n == 0:
            branch_nodes.pop(int(chosen), None)
            continue
        next_graph = _build_graph(next_sk)
        reward = float(
            calculate_step_reward(
                old_state=_state_payload(point_cloud, skeleton, graph, initial_branch_count),
                new_state=_state_payload(next_pc, next_sk, next_graph, initial_branch_count),
                action_branch=int(chosen),
                is_stop_action=False,
            )
        )

        snapshots.append({
            "step": step,
            "action": int(chosen),
            "reward": reward,
            "pruning_ratio_before": ratio,
            "point_cloud_before": point_cloud.copy(),
            "skeleton_before": skeleton.copy(),
            "removed_branch_ids": set(int(b) for b in removed_set),
        })

        pruned_ids.update(removed_set)
        cumulative += reward
        point_cloud = next_pc
        skeleton = next_sk
        point_branch_ids = next_pbids
        graph = next_graph
        branch_nodes = _branch_to_nodes(graph)
        if graph.number_of_nodes() > 0:
            root_branch = _root_branch_id(graph)

    return {
        "tree_name": tree_dir.name,
        "snapshots": snapshots,
        "total_reward": cumulative,
        "final_pruning_ratio": len(pruned_ids) / max(1, initial_branch_count),
    }


def _pick_step_indices(num: int, n: int) -> list[int]:
    if num <= 0:
        return []
    if num <= n:
        return list(range(num))
    return [int(round(i * (num - 1) / (n - 1))) for i in range(n)]


def _scatter_step(ax, point_cloud: np.ndarray, skel: np.ndarray, removed_ids: set[int]) -> None:
    """Main view: grey remaining cloud + red pruned region. A zoom inset on
    the upper-right corner enlarges the pruned region so it's identifiable
    even when the branch is small in the global view."""
    if point_cloud.size == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center", transform=ax.transAxes)
        return
    pc_branch_ids = _point_branch_ids(point_cloud, skel)
    if removed_ids:
        removed_ids_arr = np.fromiter(removed_ids, dtype=np.int64)
        red_pc_mask = np.isin(pc_branch_ids, removed_ids_arr)
        red_sk_mask = np.isin(skel[:, 5].astype(np.int64), removed_ids_arr)
    else:
        red_pc_mask = np.zeros(len(point_cloud), dtype=bool)
        red_sk_mask = np.zeros(len(skel), dtype=bool)
    grey_pts = point_cloud[~red_pc_mask]
    red_pc_pts = point_cloud[red_pc_mask]
    red_sk_pts = skel[red_sk_mask][:, [0, 2]] if red_sk_mask.any() else None

    if grey_pts.size:
        ax.scatter(grey_pts[:, 0], grey_pts[:, 2], s=GREY_SIZE, c=[GREY_RGBA], linewidths=0)
    if red_pc_pts.size:
        ax.scatter(red_pc_pts[:, 0], red_pc_pts[:, 2], s=RED_PC_SIZE, c=[RED_PC_RGBA], linewidths=0)
    if red_sk_pts is not None and len(red_sk_pts):
        ax.scatter(red_sk_pts[:, 0], red_sk_pts[:, 1], s=RED_SK_SIZE, c=[RED_SK_RGBA], linewidths=0, zorder=5)

    if red_pc_pts.size == 0 and (red_sk_pts is None or len(red_sk_pts) == 0):
        return

    if red_pc_pts.size:
        focus_pts = red_pc_pts[:, [0, 2]]
    else:
        focus_pts = red_sk_pts
    fx_min, fx_max = float(focus_pts[:, 0].min()), float(focus_pts[:, 0].max())
    fz_min, fz_max = float(focus_pts[:, 1].min()), float(focus_pts[:, 1].max())
    span = max(fx_max - fx_min, fz_max - fz_min, 0.4)
    cx, cz = 0.5 * (fx_min + fx_max), 0.5 * (fz_min + fz_max)
    pad = span * 1.4
    ix_lo, ix_hi = cx - pad, cx + pad
    iz_lo, iz_hi = cz - pad, cz + pad

    inset = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
    in_box = (
        (point_cloud[:, 0] >= ix_lo) & (point_cloud[:, 0] <= ix_hi)
        & (point_cloud[:, 2] >= iz_lo) & (point_cloud[:, 2] <= iz_hi)
    )
    grey_in = point_cloud[in_box & ~red_pc_mask]
    red_in = point_cloud[in_box & red_pc_mask]
    if grey_in.size:
        inset.scatter(grey_in[:, 0], grey_in[:, 2], s=1.0, c=[GREY_RGBA], linewidths=0)
    if red_in.size:
        inset.scatter(red_in[:, 0], red_in[:, 2], s=14.0, c=[RED_PC_RGBA], linewidths=0)
    if red_sk_pts is not None and len(red_sk_pts):
        inset.scatter(red_sk_pts[:, 0], red_sk_pts[:, 1], s=18.0, c=[RED_SK_RGBA], linewidths=0, zorder=5)
    inset.set_xlim(ix_lo, ix_hi)
    inset.set_ylim(iz_lo, iz_hi)
    inset.set_xticks([])
    inset.set_yticks([])
    for s in inset.spines.values():
        s.set_edgecolor("0.5")
        s.set_linewidth(0.8)
    ax.indicate_inset_zoom(inset, edgecolor="0.5", linewidth=0.8, alpha=0.7)


def plot_learned_rollout(rollout: dict[str, Any], out_path: Path, n_steps: int = N_STEPS_TO_SHOW) -> None:
    snaps = rollout["snapshots"]
    if not snaps:
        raise RuntimeError(f"No snapshots produced for {rollout['tree_name']}")

    idxs = _pick_step_indices(len(snaps), n_steps)

    pc0 = snaps[0]["point_cloud_before"]
    pad = 0.5
    x_min, x_max = float(pc0[:, 0].min()) - pad, float(pc0[:, 0].max()) + pad
    z_min, z_max = float(pc0[:, 2].min()) - pad, float(pc0[:, 2].max()) + pad

    fig, axes = plt.subplots(1, n_steps, figsize=(3.6 * n_steps, 4.0), squeeze=False)
    fig.suptitle(
        f"Learned-policy pruning rollout for {rollout['tree_name']}  "
        f"(G={rollout['total_reward']:.2f}, ratio={rollout['final_pruning_ratio']:.3f})",
        fontsize=14, y=1.02,
    )

    for col_idx, t in enumerate(idxs):
        snap = snaps[t]
        ax = axes[0, col_idx]
        _scatter_step(ax, snap["point_cloud_before"], snap["skeleton_before"], snap["removed_branch_ids"])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"t={snap['step']}, a={snap['action']}\nr={snap['reward']:.3f}", fontsize=10)
        ax.set_xlabel("X")
        if col_idx == 0:
            ax.set_ylabel("Learned trajectory", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote: {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", default="Lemon00_01")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-steps", type=int, default=N_STEPS_TO_SHOW)
    parser.add_argument("--max-steps", type=int, default=MAX_BENCHMARK_STEPS)
    args = parser.parse_args()

    tree_dir = Path(args.data_root) / args.tree
    if not tree_dir.exists():
        raise FileNotFoundError(tree_dir)
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"learned_rollout_{args.tree}.png"

    rl_model, ckpt = _load_rl_policy(ckpt_path)
    print(f"Loaded {ckpt_path} | kind={rl_model['kind']} | epoch={ckpt.get('epoch', 'NA')} | device={RB_DEVICE}")

    rollout = rollout_with_snapshots(tree_dir, rl_model, max_steps=args.max_steps)
    print(
        f"Rollout {args.tree}: steps={len(rollout['snapshots'])} "
        f"total_reward={rollout['total_reward']:.3f} ratio={rollout['final_pruning_ratio']:.3f}"
    )
    plot_learned_rollout(rollout, out_path, n_steps=args.n_steps)


if __name__ == "__main__":
    main()
