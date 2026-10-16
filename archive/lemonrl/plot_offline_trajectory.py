"""
Recreate the offline-trajectory example figure (trajectory_example_<Tree>.png).

Reads a Random trajectory pickle and an Expert trajectory pickle for the same
tree and draws a 2 x N_STEPS grid of X-Z scatter plots. At each shown step the
remaining tree skeleton/point cloud is drawn in grey and the branch about to be
pruned (i.e. the branch removed between step t and step t+1) is highlighted in
red.

This script is intended to run inside the conda env that pickled the .pkl
files. On this machine that is the 'rl' env (Python 3.9, numpy 2.0):

    & D:\\PROJECTS\\deeplearning\\envs\\rl\\python.exe plot_offline_trajectory.py

It does NOT need PyTorch or the model checkpoint.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import pickle
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_trajectories import _point_branch_ids


TRAJ_DIR = Path("data/trajectories")
DEFAULT_OUT_DIR = Path("../论文/figures")
N_STEPS_TO_SHOW = 4
GREY_RGBA = (0.35, 0.35, 0.35, 0.45)
RED_PC_RGBA = (0.85, 0.10, 0.10, 0.85)
RED_SK_RGBA = (0.85, 0.05, 0.05, 1.00)
GREY_SIZE = 0.25
RED_PC_SIZE = 4.0
RED_SK_SIZE = 6.0


def _pick_step_indices(num_steps: int, n: int) -> list[int]:
    """Choose n roughly evenly spaced step indices in [0, num_steps-1]."""
    if num_steps <= 0:
        return []
    if num_steps <= n:
        return list(range(num_steps))
    return [int(round(i * (num_steps - 1) / (n - 1))) for i in range(n)]


def _removed_branch_ids_between(skel_before: np.ndarray, skel_after: np.ndarray | None) -> set[int]:
    """Branch ids that exist in skel_before but not skel_after (or the empty set)."""
    before = set(int(b) for b in skel_before[:, 5])
    after = set(int(b) for b in skel_after[:, 5]) if skel_after is not None else set()
    return before - after


def _scatter_step(ax, point_cloud: np.ndarray, skel: np.ndarray, removed_ids: set[int]) -> None:
    """Main view: grey remaining cloud + red pruned region. A zoom inset on
    the upper-right corner enlarges the pruned branch so it is clear which
    branch is being cut, even when its global footprint is small."""
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

    focus_pts = red_pc_pts[:, [0, 2]] if red_pc_pts.size else red_sk_pts
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


def _axis_bounds(traj: dict, pad: float = 0.5) -> tuple[float, float, float, float]:
    """Bounding box in the X-Z plane over the *first* point cloud, padded."""
    pc = traj["point_clouds"][0]
    x_min, x_max = float(pc[:, 0].min()), float(pc[:, 0].max())
    z_min, z_max = float(pc[:, 2].min()), float(pc[:, 2].max())
    return x_min - pad, x_max + pad, z_min - pad, z_max + pad


MODE_TITLES = {
    "both": "Offline pruning trajectory example",
    "random": "Random pruning trajectory",
    "expert": "Expert pruning trajectory",
}


def plot_trajectory_pair(
    tree_name: str,
    random_pkl: Path | None,
    expert_pkl: Path | None,
    out_path: Path,
    n_steps: int = N_STEPS_TO_SHOW,
    mode: str = "both",
) -> None:
    rows: list[tuple[str, dict]] = []
    if mode in ("both", "random"):
        with open(random_pkl, "rb") as f:
            rnd = pickle.load(f)
        rows.append(("Random trajectory", rnd))
    if mode in ("both", "expert"):
        with open(expert_pkl, "rb") as f:
            exp = pickle.load(f)
        rows.append(("Expert trajectory", exp))
    if not rows:
        raise ValueError(f"Invalid mode: {mode}")

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_steps, figsize=(3.6 * n_steps, 3.6 * n_rows + 0.6), squeeze=False)
    fig.suptitle(
        f"{MODE_TITLES.get(mode, MODE_TITLES['both'])} for {tree_name}",
        fontsize=15, y=0.995,
    )

    for row_idx, (row_label, traj) in enumerate(rows):
        skeletons: Sequence[np.ndarray] = traj["skeletons"]
        point_clouds: Sequence[np.ndarray] = traj["point_clouds"]
        actions: Sequence[int] = traj["actions"]
        rewards: Sequence[float] = traj["rewards"]
        returns_to_go: Sequence[float] = traj["returns_to_go"]
        stop_action = int(traj["metadata"].get("stop_action", -1))
        valid_step_idxs = [t for t in range(len(actions)) if int(actions[t]) != stop_action]
        idxs = [valid_step_idxs[i] for i in _pick_step_indices(len(valid_step_idxs), n_steps)]

        final_pruning_ratio = (
            traj["metadata"].get("target_pruning_ratio")
            or traj["metadata"].get("max_pruning_ratio")
            or 0.0
        )
        cumulative_return = float(returns_to_go[0]) if returns_to_go else 0.0
        row_subtitle = f"{row_label}\nG={cumulative_return:.2f}, ratio={final_pruning_ratio:.3f}"

        x_min, x_max, z_min, z_max = _axis_bounds(traj)

        for col_idx, t in enumerate(idxs):
            ax = axes[row_idx, col_idx]
            skel = skeletons[t]
            pc = point_clouds[t]
            skel_next = skeletons[t + 1] if (t + 1) < len(skeletons) else None
            removed_ids = _removed_branch_ids_between(skel, skel_next)
            _scatter_step(ax, pc, skel, removed_ids)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(z_min, z_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"t={t}, a={int(actions[t])}\nr={float(rewards[t]):.3f}", fontsize=10)
            if row_idx == n_rows - 1:
                ax.set_xlabel("X")
            if col_idx == 0:
                ax.set_ylabel(row_subtitle, fontsize=10)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote: {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", default="Lemon00_01", help="Tree name (e.g. Lemon00_01)")
    parser.add_argument("--mode", default="both", choices=["both", "random", "expert"],
                        help="Which trajectory rows to draw")
    parser.add_argument("--random-pkl", default=None, help="Optional explicit path to the random trajectory pkl")
    parser.add_argument("--expert-pkl", default=None, help="Optional explicit path to the expert trajectory pkl")
    parser.add_argument("--random-episode", type=int, default=1, help="Random trajectory episode index (default 1)")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--n-steps", type=int, default=N_STEPS_TO_SHOW, help="Number of step snapshots per row")
    args = parser.parse_args()

    random_pkl = (
        Path(args.random_pkl)
        if args.random_pkl
        else TRAJ_DIR / f"random_traj_{args.tree}_{args.random_episode}.pkl"
    )
    expert_pkl = (
        Path(args.expert_pkl)
        if args.expert_pkl
        else TRAJ_DIR / f"expert_traj_{args.tree}.pkl"
    )
    default_basename = {
        "both": f"trajectory_example_{args.tree}.png",
        "random": f"random_rollout_{args.tree}.png",
        "expert": f"expert_rollout_{args.tree}.png",
    }[args.mode]
    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / default_basename

    if args.mode in ("both", "random") and not random_pkl.exists():
        raise FileNotFoundError(random_pkl)
    if args.mode in ("both", "expert") and not expert_pkl.exists():
        raise FileNotFoundError(expert_pkl)

    plot_trajectory_pair(args.tree, random_pkl, expert_pkl, out_path, n_steps=args.n_steps, mode=args.mode)


if __name__ == "__main__":
    main()
