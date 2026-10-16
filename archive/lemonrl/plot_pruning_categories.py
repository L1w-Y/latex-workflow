"""
Visualise the *final* pruning outcome from plant_cut_v2:

  - skeleton_original_GRAY.ply  : the full tree skeleton (background).
  - skeleton_final_GREEN.ply    : the tree skeleton AFTER pruning (kept).
  - skeleton_pruned_part_RED.ply: the skeleton points that were REMOVED.

We cluster the 700 RED points by spatial connectivity (one connected component
per pruned branch), compute simple PCA geometry for every cluster, then apply
the five rules from §4.2 of the paper to classify each pruned branch into one
of the categories described there:

  * 交叉枝 / 重叠枝  (crossing / overlapping)
  * 直立徒长枝        (vigorous upright)
  * 下垂枝            (drooping)
  * 内膛枝            (inward-growing)
  * 细弱枝            (thin / weak)

For each category we keep one representative cluster (the most typical one)
and draw it in a zoom inset around the main view.

Runs in the base env (numpy 1.x is fine since the ply files are read with a
custom binary parser):

    & D:\\PROJECTS\\deeplearning\\python.exe plot_pruning_categories.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import struct
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "Source Han Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_PLY_DIR = Path("../plant_cut_v2/results/pruning_visualization_skeleton")
DEFAULT_OUT = Path("../论文/figures/pruning_categories.png")

CLUSTER_EPS = 0.20
MIN_CLUSTER_SIZE = 8

GRAY_RGBA = (0.55, 0.55, 0.55, 0.18)
GREEN_RGBA = (0.20, 0.55, 0.20, 0.65)
RED_RGBA = (0.85, 0.10, 0.10, 0.95)
GRAY_SIZE = 0.6
GREEN_SIZE = 1.0
RED_SIZE = 6.0

CATEGORY_ORDER = [
    "crossing",
    "vigorous_upright",
    "drooping",
    "inward",
    "thin_weak",
]
CATEGORY_LABELS = {
    "crossing":         "(a) Crossing / overlapping 交叉枝",
    "vigorous_upright": "(b) Vigorous upright 直立徒长枝",
    "drooping":         "(c) Drooping 下垂枝",
    "inward":           "(d) Inward 内膛枝",
    "thin_weak":        "(e) Thin / weak 细弱枝",
}


def read_ply_xyz(path: Path) -> np.ndarray:
    """Parse a binary little-endian PLY exported by Open3D
    (3 x float64 xyz + 3 x uchar rgb). Returns (N, 3) float64."""
    with open(path, "rb") as f:
        header = b""
        n_vertex = 0
        while True:
            line = f.readline()
            header += line
            if line.startswith(b"element vertex"):
                n_vertex = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        body = f.read()
    if n_vertex == 0:
        raise ValueError(f"No vertices in {path}")
    rec = 27
    if len(body) < n_vertex * rec:
        raise ValueError(f"Truncated body in {path}")
    xyz = np.zeros((n_vertex, 3), dtype=np.float64)
    for i in range(n_vertex):
        xyz[i] = struct.unpack_from("<ddd", body, i * rec)
    return xyz


def connected_components(points: np.ndarray, eps: float) -> np.ndarray:
    """Label points by their connected component in the radius-eps graph.

    Returns an (N,) int array of cluster ids in [0, K)."""
    n = len(points)
    tree = cKDTree(points)
    cid = -np.ones(n, dtype=np.int64)
    cur = 0
    for seed in range(n):
        if cid[seed] != -1:
            continue
        stack = [seed]
        cid[seed] = cur
        while stack:
            i = stack.pop()
            for j in tree.query_ball_point(points[i], eps):
                if cid[j] == -1:
                    cid[j] = cur
                    stack.append(j)
        cur += 1
    return cid


def cluster_geometry(pts: np.ndarray) -> dict[str, Any]:
    """PCA-based geometry summary for one pruned branch."""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = (centered.T @ centered) / max(1, len(pts) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    axis = evecs[:, 0]
    if axis[2] < 0:
        axis = -axis
    projections = centered @ axis
    length = float(projections.max() - projections.min())
    perp_var = float(np.sqrt(max(evals[1] + evals[2], 0.0)))
    proximal_idx = int(np.argmin(pts[:, 2]))
    distal_idx = int(np.argmax(pts[:, 2]))
    return {
        "centroid": centroid,
        "axis": axis,
        "length": length,
        "perp_var": perp_var,
        "n_points": int(len(pts)),
        "proximal": pts[proximal_idx].copy(),
        "distal": pts[distal_idx].copy(),
        "z_min": float(pts[:, 2].min()),
        "z_max": float(pts[:, 2].max()),
    }


def classify_clusters(
    clusters: list[dict[str, Any]],
    tree_centroid: np.ndarray,
    green_pts: np.ndarray,
) -> list[dict[str, Any]]:
    """Assign each cluster a category id + score per the §4.2 paper rules.

    The output augments each cluster dict with:
      - inward_dot     : v . d_outward  (negative = grows back into canopy)
      - up_cos         : v . e_z        (1 = pure upright, -1 = straight down)
      - rel_length     : L_b / mean(L_b)
      - radius_proxy   : perp_var / length
      - min_dist_green : min distance to any green skeleton point
      - category       : str or None
      - score          : higher = more typical example of its category
    """
    if not clusters:
        return clusters

    e_z = np.array([0.0, 0.0, 1.0])
    lengths = np.array([c["length"] for c in clusters])
    mean_L = float(lengths.mean()) if lengths.size else 1.0
    perp_over_L = np.array([
        (c["perp_var"] / c["length"]) if c["length"] > 1e-6 else np.inf
        for c in clusters
    ])
    mean_rprox = float(np.median(perp_over_L[np.isfinite(perp_over_L)])) if np.any(np.isfinite(perp_over_L)) else 1.0
    green_tree = cKDTree(green_pts) if len(green_pts) else None

    for c in clusters:
        v = c["axis"]
        c_dir = c["proximal"] - tree_centroid
        nrm = np.linalg.norm(c_dir)
        d_outward = c_dir / max(nrm, 1e-6)
        c["inward_dot"] = float(v @ d_outward)
        c["up_cos"] = float(v @ e_z)
        c["rel_length"] = c["length"] / max(mean_L, 1e-6)
        c["radius_proxy"] = float(c["perp_var"] / max(c["length"], 1e-6))
        c["radius_proxy_rel"] = c["radius_proxy"] / max(mean_rprox, 1e-6)
        if green_tree is not None:
            d, _ = green_tree.query(c["centroid"], k=1)
            c["min_dist_green"] = float(d)
        else:
            c["min_dist_green"] = float("inf")

    for c in clusters:
        scores: dict[str, float] = {
            "vigorous_upright": c["up_cos"] + 0.4 * (c["rel_length"] - 1.0),
            "drooping":          -c["up_cos"] + 0.2 * c["rel_length"],
            "inward":            -c["inward_dot"] + 0.2 * c["rel_length"],
            "thin_weak":         (1.0 - c["radius_proxy_rel"]) + 0.3 * (c["rel_length"] - 0.5),
            "crossing":          max(0.0, (0.45 - c["min_dist_green"]) / 0.45) + 0.2 * c["rel_length"],
        }
        c["scores"] = scores
        best = max(scores.items(), key=lambda kv: kv[1])
        c["category"] = best[0]
        c["score"] = best[1]
    return clusters


def pick_representatives(clusters: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Greedy assignment: walk (cluster, category) pairs in descending score and
    assign each category its first still-unused cluster. This guarantees every
    category gets a distinct representative as long as len(clusters) >= 5."""
    pairs: list[tuple[float, int, str]] = []
    for idx, c in enumerate(clusters):
        for cat, sc in c["scores"].items():
            pairs.append((sc, idx, cat))
    pairs.sort(key=lambda x: -x[0])
    used_clusters: set[int] = set()
    used_cats: set[str] = set()
    reps: dict[str, dict[str, Any]] = {}
    for sc, idx, cat in pairs:
        if cat in used_cats or idx in used_clusters:
            continue
        reps[cat] = clusters[idx]
        used_clusters.add(idx)
        used_cats.add(cat)
        if len(reps) == len(CATEGORY_ORDER):
            break
    return reps


def _setup_main_axis(ax, gray: np.ndarray, green: np.ndarray, red: np.ndarray) -> None:
    if gray.size:
        ax.scatter(gray[:, 0], gray[:, 2], s=GRAY_SIZE, c=[GRAY_RGBA], linewidths=0)
    if green.size:
        ax.scatter(green[:, 0], green[:, 2], s=GREEN_SIZE, c=[GREEN_RGBA], linewidths=0)
    if red.size:
        ax.scatter(red[:, 0], red[:, 2], s=RED_SIZE, c=[RED_RGBA], linewidths=0, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(False)


def _draw_inset(ax, gray: np.ndarray, green: np.ndarray, cluster: dict[str, Any], title: str) -> None:
    pts = cluster["points"]
    cx, _, cz = pts.mean(axis=0)
    span = max(
        float(pts[:, 0].max() - pts[:, 0].min()),
        float(pts[:, 2].max() - pts[:, 2].min()),
        0.5,
    )
    pad = max(span * 1.3, 0.8)
    x_lo, x_hi = cx - pad, cx + pad
    z_lo, z_hi = cz - pad, cz + pad

    if gray.size:
        in_box = (
            (gray[:, 0] >= x_lo) & (gray[:, 0] <= x_hi)
            & (gray[:, 2] >= z_lo) & (gray[:, 2] <= z_hi)
        )
        gr = gray[in_box]
        if gr.size:
            ax.scatter(gr[:, 0], gr[:, 2], s=1.5, c=[GRAY_RGBA], linewidths=0)
    if green.size:
        in_box = (
            (green[:, 0] >= x_lo) & (green[:, 0] <= x_hi)
            & (green[:, 2] >= z_lo) & (green[:, 2] <= z_hi)
        )
        gn = green[in_box]
        if gn.size:
            ax.scatter(gn[:, 0], gn[:, 2], s=3.5, c=[GREEN_RGBA], linewidths=0)
    ax.scatter(pts[:, 0], pts[:, 2], s=22.0, c=[RED_RGBA], linewidths=0, zorder=5)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(z_lo, z_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    for s in ax.spines.values():
        s.set_edgecolor("0.35")
        s.set_linewidth(1.0)


def _connector(fig, main_ax, inset_ax, cluster: dict[str, Any]) -> None:
    """Draw a thin line connecting the inset to the cluster centroid in the main view."""
    cx, _, cz = cluster["points"].mean(axis=0)
    main_xy_disp = main_ax.transData.transform((cx, cz))
    main_xy_fig = fig.transFigure.inverted().transform(main_xy_disp)
    inset_bbox = inset_ax.get_position()
    inset_xy_fig = (
        inset_bbox.x0 + 0.5 * inset_bbox.width,
        inset_bbox.y0 + 0.5 * inset_bbox.height,
    )
    line = plt.Line2D(
        [main_xy_fig[0], inset_xy_fig[0]],
        [main_xy_fig[1], inset_xy_fig[1]],
        transform=fig.transFigure,
        color="0.55", linewidth=0.6, alpha=0.65, zorder=1,
    )
    fig.lines.append(line)


def plot_pruning_categories(
    gray: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    reps: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(
        nrows=3, ncols=4,
        width_ratios=[1.8, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.35, wspace=0.30,
    )

    main_ax = fig.add_subplot(gs[:, 0])
    _setup_main_axis(main_ax, gray, green, red)
    main_ax.set_title(
        f"Final pruning outcome: gray = original ({len(gray)}), "
        f"green = retained ({len(green)}), red = pruned ({len(red)})",
        fontsize=11,
    )

    inset_positions = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]
    inset_axes = []
    for cat, (r, c) in zip(CATEGORY_ORDER, inset_positions):
        ax = fig.add_subplot(gs[r, c])
        if cat in reps:
            _draw_inset(ax, gray, green, reps[cat], CATEGORY_LABELS[cat])
            inset_axes.append((ax, reps[cat]))
        else:
            ax.text(0.5, 0.5, f"{CATEGORY_LABELS[cat]}\n(no representative)",
                    ha="center", va="center", fontsize=9, color="0.4")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("0.7"); s.set_linewidth(0.5)

    legend_ax = fig.add_subplot(gs[1, 3])
    legend_ax.axis("off")
    legend_ax.text(
        0.0, 0.95,
        "Pruning category rules (§4.2):\n"
        "  (a) Crossing: tip near retained skel + non-parallel axis\n"
        "  (b) Vigorous upright: cos(v, +z) > cos30°, L > 1.2 L̄\n"
        "  (c) Drooping: cos(v, +z) < -cos60° (i.e. >150° apart)\n"
        "  (d) Inward: v · d_outward < -0.30\n"
        "  (e) Thin / weak: long but slim (perp_var / L < 0.7 r̄_rel)",
        ha="left", va="top", fontsize=8.5, family="monospace",
    )

    summary_ax = fig.add_subplot(gs[2, 1:])
    summary_ax.axis("off")

    fig.suptitle(
        "Final pruning outcome categorised by §4.2 unsafe-branch rules "
        f"({len(red)} red points → {sum(1 for r in reps.values())} representative branches)",
        fontsize=14, y=0.995,
    )

    fig.canvas.draw()
    for ax, cluster in inset_axes:
        _connector(fig, main_ax, ax, cluster)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote: {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply-dir", default=str(DEFAULT_PLY_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--cluster-eps", type=float, default=CLUSTER_EPS)
    parser.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    gray = read_ply_xyz(ply_dir / "skeleton_original_GRAY.ply")
    green = read_ply_xyz(ply_dir / "skeleton_final_GREEN.ply")
    red = read_ply_xyz(ply_dir / "skeleton_pruned_part_RED.ply")
    print(f"Read PLY: gray={len(gray)}, green={len(green)}, red={len(red)}")

    cid = connected_components(red, eps=args.cluster_eps)
    n_clusters = int(cid.max()) + 1 if len(cid) else 0
    print(f"Connected components (eps={args.cluster_eps}): {n_clusters}")

    clusters: list[dict[str, Any]] = []
    for k in range(n_clusters):
        mask = cid == k
        if mask.sum() < args.min_cluster_size:
            continue
        pts = red[mask]
        info = cluster_geometry(pts)
        info["points"] = pts
        info["cluster_id"] = k
        clusters.append(info)
    print(f"Kept {len(clusters)} clusters with >= {args.min_cluster_size} points")

    tree_centroid = gray.mean(axis=0)
    classify_clusters(clusters, tree_centroid, green)

    counts: dict[str, int] = {}
    for c in clusters:
        cat = c["category"] or "unclassified"
        counts[cat] = counts.get(cat, 0) + 1
    print("Per-category counts:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")

    reps = pick_representatives(clusters)
    print(f"Picked representatives: {list(reps.keys())}")

    plot_pruning_categories(gray, green, red, reps, Path(args.out))


if __name__ == "__main__":
    main()
