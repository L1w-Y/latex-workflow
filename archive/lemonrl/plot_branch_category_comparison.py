from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from generate_trajectories import (
    MAX_PRUNING_RATIO,
    _build_graph,
    _branch_to_nodes,
    _read_skeleton,
    _removed_branch_ids_for_action,
    _root_branch_id,
    find_worst_branch,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


CATEGORY_ORDER = ["crossing", "vigorous_upright", "drooping", "inward", "thin_weak"]
CATEGORY_LABELS = {
    "crossing": "Crossing / overlap",
    "vigorous_upright": "Vigorous upright",
    "drooping": "Drooping",
    "inward": "Inward",
    "thin_weak": "Thin / weak",
}
CATEGORY_COLORS = {
    "crossing": "#e41a1c",
    "vigorous_upright": "#ff7f00",
    "drooping": "#377eb8",
    "inward": "#984ea3",
    "thin_weak": "#4daf4a",
}


def simulate_greedy_pruning(skeleton: np.ndarray, max_steps: int, greedy_threshold: float) -> dict[str, Any]:
    working = skeleton.copy()
    initial_graph = _build_graph(working)
    initial_branch_nodes = _branch_to_nodes(initial_graph)
    initial_root = _root_branch_id(initial_graph)
    initial_cuttable = [b for b in initial_branch_nodes.keys() if int(b) >= 0 and int(b) != int(initial_root)]
    initial_count = len(set(initial_cuttable))

    if initial_count <= 0:
        return {
            "final_skeleton": working,
            "removed_branch_ids": set(),
            "step_records": [],
            "initial_branch_count": 0,
            "max_allowed_cuts": 0,
        }

    max_allowed_cuts = max(1, int(np.floor(MAX_PRUNING_RATIO * initial_count + 1e-8)))
    removed_branch_ids: set[int] = set()
    step_records: list[dict[str, Any]] = []

    for step in range(max_steps):
        graph = _build_graph(working)
        if graph.number_of_nodes() == 0:
            break
        branch_nodes = _branch_to_nodes(graph)
        root_branch = _root_branch_id(graph)
        valid = [b for b in branch_nodes.keys() if int(b) >= 0 and int(b) != int(root_branch)]
        if (not valid) or len(removed_branch_ids) >= max_allowed_cuts:
            break

        chosen, score = find_worst_branch(graph, branch_nodes, root_branch)
        if chosen is None or float(score) < float(greedy_threshold):
            break

        to_remove = _removed_branch_ids_for_action(graph, int(chosen), branch_nodes)
        if not to_remove:
            break

        projected = len(removed_branch_ids.union(to_remove))
        if projected > max_allowed_cuts:
            break

        keep_mask = ~np.isin(working[:, 5].astype(np.int64), np.array(sorted(to_remove), dtype=np.int64))
        new_working = working[keep_mask]
        if len(new_working) == len(working):
            break

        removed_branch_ids.update(int(x) for x in to_remove)
        step_records.append(
            {
                "step": int(step + 1),
                "action_branch": int(chosen),
                "heuristic_score": float(score),
                "removed_ids": sorted(int(x) for x in to_remove),
                "remaining_points": int(len(new_working)),
            }
        )
        working = new_working

    return {
        "final_skeleton": working,
        "removed_branch_ids": removed_branch_ids,
        "step_records": step_records,
        "initial_branch_count": int(initial_count),
        "max_allowed_cuts": int(max_allowed_cuts),
    }


def extract_branch_features(initial_skeleton: np.ndarray, removed_ids: set[int]) -> list[dict[str, Any]]:
    if not removed_ids:
        return []
    out: list[dict[str, Any]] = []
    all_branch_ids = np.unique(initial_skeleton[:, 5].astype(np.int64))
    all_centers = {}
    for bid in all_branch_ids:
        pts = initial_skeleton[initial_skeleton[:, 5].astype(np.int64) == bid, :3]
        if len(pts) > 0:
            all_centers[int(bid)] = pts.mean(axis=0)
    global_center = initial_skeleton[:, :3].mean(axis=0)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for bid in sorted(int(x) for x in removed_ids):
        pts = initial_skeleton[initial_skeleton[:, 5].astype(np.int64) == bid, :3]
        if len(pts) < 4:
            continue
        center = pts.mean(axis=0)
        centered = pts - center
        cov = centered.T @ centered / max(1, len(pts) - 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        axis = evecs[:, order[0]]
        if axis[2] < 0:
            axis = -axis
        proj = centered @ axis
        length = float(proj.max() - proj.min())
        thickness = float(np.sqrt(max(evals[1] + evals[2], 1e-12)))
        up_cos = float(axis @ z_axis)
        outward = center - global_center
        outward /= max(np.linalg.norm(outward), 1e-9)
        inward_dot = float(axis @ outward)
        nearest_dist = np.inf
        for other_bid, other_center in all_centers.items():
            if other_bid == bid:
                continue
            d = float(np.linalg.norm(other_center - center))
            if d < nearest_dist:
                nearest_dist = d
        out.append(
            {
                "branch_id": int(bid),
                "points": pts.copy(),
                "center": center,
                "length": length,
                "thickness": thickness,
                "up_cos": up_cos,
                "inward_dot": inward_dot,
                "nearest_dist": float(nearest_dist),
            }
        )
    return out


def classify_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not features:
        return []
    lengths = np.array([f["length"] for f in features], dtype=np.float64)
    thickness_ratio = np.array(
        [f["thickness"] / max(f["length"], 1e-9) for f in features], dtype=np.float64
    )
    mean_len = float(max(lengths.mean(), 1e-9))
    med_ratio = float(max(np.median(thickness_ratio), 1e-9))

    for f in features:
        rel_len = f["length"] / mean_len
        rel_thin = (f["thickness"] / max(f["length"], 1e-9)) / med_ratio
        scores = {
            "crossing": max(0.0, (0.45 - f["nearest_dist"]) / 0.45) + 0.20 * rel_len,
            "vigorous_upright": f["up_cos"] + 0.35 * (rel_len - 1.0),
            "drooping": -f["up_cos"] + 0.20 * rel_len,
            "inward": -f["inward_dot"] + 0.15 * rel_len,
            "thin_weak": (1.0 - rel_thin) + 0.30 * (rel_len - 0.5),
        }
        cat, score = max(scores.items(), key=lambda kv: kv[1])
        f["category"] = cat
        f["score"] = float(score)
        f["scores"] = scores
    return features


def pick_representatives(features: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    reps: dict[str, dict[str, Any]] = {}
    for cat in CATEGORY_ORDER:
        cands = [f for f in features if f.get("category") == cat]
        if not cands:
            continue
        cands.sort(key=lambda x: x["score"], reverse=True)
        reps[cat] = cands[0]
    return reps


def _draw_global_before(ax, skeleton: np.ndarray, features: list[dict[str, Any]], reps: dict[str, dict[str, Any]]) -> None:
    ax.scatter(skeleton[:, 0], skeleton[:, 2], s=0.5, c="#8a8a8a", alpha=0.18, linewidths=0)
    for f in features:
        p = f["points"][:, [0, 2]]
        color = CATEGORY_COLORS.get(f["category"], "#d62728")
        ax.scatter(p[:, 0], p[:, 1], s=6.0, c=color, alpha=0.95, linewidths=0, zorder=5)
    for _, f in reps.items():
        p = f["points"][:, [0, 2]]
        x0, x1 = float(np.min(p[:, 0])), float(np.max(p[:, 0]))
        z0, z1 = float(np.min(p[:, 1])), float(np.max(p[:, 1]))
        pad = max(max(x1 - x0, z1 - z0), 0.20) * 0.85
        rect = Rectangle((x0 - pad, z0 - pad), (x1 - x0) + 2 * pad, (z1 - z0) + 2 * pad,
                         fill=False, edgecolor=CATEGORY_COLORS[f["category"]], linewidth=1.1, alpha=0.9)
        ax.add_patch(rect)
    ax.set_title("Before pruning (colored = pruned branches)", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_global_after(ax, final_skeleton: np.ndarray) -> None:
    if len(final_skeleton):
        ax.scatter(final_skeleton[:, 0], final_skeleton[:, 2], s=0.7, c="#2ca25f", alpha=0.42, linewidths=0)
    ax.set_title("After pruning (retained skeleton)", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_zoom(ax, skeleton: np.ndarray, f: dict[str, Any]) -> None:
    pts = f["points"][:, [0, 2]]
    cx = float(np.mean(pts[:, 0]))
    cz = float(np.mean(pts[:, 1]))
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 0.40)
    pad = span * 1.6
    x0, x1 = cx - pad, cx + pad
    z0, z1 = cz - pad, cz + pad

    in_box = (
        (skeleton[:, 0] >= x0) & (skeleton[:, 0] <= x1) &
        (skeleton[:, 2] >= z0) & (skeleton[:, 2] <= z1)
    )
    bg = skeleton[in_box][:, [0, 2]]
    if len(bg):
        ax.scatter(bg[:, 0], bg[:, 1], s=1.4, c="#9a9a9a", alpha=0.23, linewidths=0)
    ax.scatter(pts[:, 0], pts[:, 1], s=18.0, c=CATEGORY_COLORS[f["category"]], alpha=0.98, linewidths=0, zorder=4)

    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{CATEGORY_LABELS[f['category']]} (id={f['branch_id']})", fontsize=9)
    for sp in ax.spines.values():
        sp.set_color("#666666")
        sp.set_linewidth(0.8)


def render_figure(
    tree_name: str,
    initial_skeleton: np.ndarray,
    final_skeleton: np.ndarray,
    features: list[dict[str, Any]],
    reps: dict[str, dict[str, Any]],
    step_records: list[dict[str, Any]],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        3, 4,
        width_ratios=[1.45, 1.45, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.25,
        hspace=0.30,
    )

    ax_before = fig.add_subplot(gs[:, 0])
    ax_after = fig.add_subplot(gs[:, 1])
    _draw_global_before(ax_before, initial_skeleton, features, reps)
    _draw_global_after(ax_after, final_skeleton)

    zoom_positions = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 2)]
    for cat, pos in zip(CATEGORY_ORDER, zoom_positions):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        f = reps.get(cat)
        if f is None:
            ax.text(0.5, 0.5, f"{CATEGORY_LABELS[cat]}\n(no sample)", ha="center", va="center", fontsize=9, color="#666666")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        _draw_zoom(ax, initial_skeleton, f)

    ax_info = fig.add_subplot(gs[2, 3])
    ax_info.axis("off")
    total_removed = len({int(f["branch_id"]) for f in features})
    cat_count = {cat: sum(1 for f in features if f["category"] == cat) for cat in CATEGORY_ORDER}
    info_lines = [
        f"Tree: {tree_name}",
        f"Pruning steps: {len(step_records)}",
        f"Pruned branches: {total_removed}",
        "",
        "Category counts:",
    ]
    for cat in CATEGORY_ORDER:
        info_lines.append(f"- {CATEGORY_LABELS[cat]}: {cat_count.get(cat, 0)}")
    ax_info.text(0.0, 1.0, "\n".join(info_lines), ha="left", va="top", fontsize=10)

    fig.suptitle("Pruning comparison with local zoom and branch-type labels", fontsize=14, y=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", default="Lemon00_09", help="Tree folder name under data root.")
    parser.add_argument("--data-root", default="./data", help="Data root containing LemonXX_YY folders.")
    parser.add_argument("--max-steps", type=int, default=120, help="Maximum greedy pruning steps.")
    parser.add_argument("--greedy-threshold", type=float, default=0.45, help="Greedy stop threshold.")
    parser.add_argument(
        "--out",
        default="../论文/figures/Tree_structure_comparison_with_categories.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    tree_dir = Path(args.data_root) / args.tree
    skel_path = tree_dir / "skeleton" / "skeleton.txt"
    if not skel_path.exists():
        raise FileNotFoundError(f"Skeleton not found: {skel_path}")

    initial_skeleton = _read_skeleton(skel_path)
    sim = simulate_greedy_pruning(
        initial_skeleton,
        max_steps=int(args.max_steps),
        greedy_threshold=float(args.greedy_threshold),
    )
    final_skeleton = sim["final_skeleton"]
    removed_ids = sim["removed_branch_ids"]
    step_records = sim["step_records"]

    features = classify_features(extract_branch_features(initial_skeleton, removed_ids))
    reps = pick_representatives(features)
    render_figure(
        tree_name=args.tree,
        initial_skeleton=initial_skeleton,
        final_skeleton=final_skeleton,
        features=features,
        reps=reps,
        step_records=step_records,
        out_path=Path(args.out),
    )

    print(
        f"Wrote figure: {Path(args.out).resolve()} | "
        f"steps={len(step_records)} | removed={len(removed_ids)} "
        f"| max_allowed={sim['max_allowed_cuts']}"
    )


if __name__ == "__main__":
    main()
