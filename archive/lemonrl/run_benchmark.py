# Auto-generated standalone runner extracted from graph_lemontree.ipynb cell 7
# Bypasses Jupyter/Cursor to avoid the OMP libiomp5md.dll conflict.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib
matplotlib.use('Agg')  # no GUI needed; figures saved to ../论文/figures/

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from gnn_rl_core import GATStateEncoder, GNNSurrogateEnv, convert_to_pyg_data
from generate_trajectories import (
    STOP_ACTION,
    MAX_PRUNING_RATIO,
    _read_point_cloud,
    _read_skeleton,
    _build_graph,
    _branch_to_nodes,
    _root_branch_id,
    _point_branch_ids,
    _remove_branch_state,
    _removed_branch_ids_for_action,
    find_worst_branch,
)
from utils import calculate_step_reward

device = globals().get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
CHECKPOINT_DIR = Path(globals().get("CHECKPOINT_DIR", Path("./checkpoints/graph_gnn")))

class GraphPruningAgent(nn.Module):
    def __init__(self, action_dim: int, state_dim: int = 128, dropout: float = 0.10) -> None:
        super().__init__()
        self.encoder = GATStateEncoder(
            node_in_dim=6,
            hidden_dim=128,
            graph_out_dim=64,
            point_out_dim=64,
            state_dim=state_dim,
            num_gat_layers=3,
            heads=4,
            dropout=dropout,
            pool="meanmax",
        )
        self.policy_head = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(state_dim, action_dim),
        )

        self.surrogate_env = GNNSurrogateEnv(
            action_dim=action_dim,
            state_dim=state_dim,
            action_emb_dim=64,
            hidden_dims=(256, 128),
            output_dim=6,
            dropout=dropout,
        )



    def forward(self, graph_batch: Batch, action_idx: torch.Tensor | None = None):
        state_vec = self.encoder(graph_batch)
        policy_logits = self.policy_head(state_vec)
        chosen_actions = policy_logits.argmax(dim=-1) if action_idx is None else action_idx
        delta_pred = self.surrogate_env(state_vec, chosen_actions)
        return state_vec, policy_logits, delta_pred





class GNNCandidateRanker(nn.Module):

    def __init__(self, candidate_dim: int, state_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = GATStateEncoder(
            node_in_dim=6,
            hidden_dim=128,
            graph_out_dim=64,
            point_out_dim=64,
            state_dim=state_dim,
            num_gat_layers=3,
            heads=4,
            dropout=dropout,
            pool="meanmax",
        )

        self.scorer = nn.Sequential(
            nn.Linear(state_dim + candidate_dim, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )



    def forward(self, graph_batch: Batch, candidate_features: torch.Tensor, candidate_mask: torch.Tensor):

        state_vec = self.encoder(graph_batch)
        k = candidate_features.size(1)
        state_expand = state_vec.unsqueeze(1).expand(-1, k, -1)
        fused = torch.cat([state_expand, candidate_features], dim=-1)
        scores = self.scorer(fused).squeeze(-1)
        scores = scores.masked_fill(~candidate_mask, -1e4)
        return scores




# 完整柠檬树数据集根目录（约 1500 棵 Lemon*/）。
# 论文 5.1 节描述的测试集（约 15%）即从此处按等距索引采样得到。
LEMON_ROOT = Path(r"D:\DownLoad\baiduyunDownload\9lemon\00lemon")
BENCHMARK_NUM_TREES = 210  # 全测试集（≈ 15% × 1400）
BENCHMARK_SEED = 42        # 保留以兼容旧调用；当前采样使用等距索引，与 seed 无关
# per_tree_reward 图保留的 4 棵代表性样本——不论等距采样是否选中，都会强制并入
CASE_STUDY_TREES = ("Lemon00_01", "Lemon00_43", "Lemon00_64", "Lemon00_75")
MAX_BENCHMARK_STEPS = 12
GREEDY_THRESHOLD = 0.45
RANK_MAX_CANDIDATES = int(globals().get("RANK_MAX_CANDIDATES", 64))
RANK_FEATURE_DIM = int(globals().get("RANK_FEATURE_DIM", 15))
RANK_DROPOUT = float(globals().get("RANK_DROPOUT", 0.15))
RANK_STOP_LOGIT_MARGIN = float(globals().get("RANK_STOP_LOGIT_MARGIN", 0.15))
RL_CKPT_PATH = CHECKPOINT_DIR / "graph_candidate_ranker_best.pt"





def _choose_pointcloud_file(tree_dir: Path) -> Path:
    for candidate in (
        tree_dir / "pointcloud" / "mergePointClouds.ply",
        tree_dir / "pointcloud" / "branchPointClouds.ply",
        tree_dir / "pointcloud" / "leafPointClouds.ply",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No point cloud found in {tree_dir / 'pointcloud'}")





def _state_payload(point_cloud: np.ndarray, skeleton_data: np.ndarray, graph, initial_branch_count: int):
    return {
        "point_cloud": np.asarray(point_cloud, dtype=np.float32),
        "skeleton_data": np.asarray(skeleton_data, dtype=np.float32),
        "graph": graph,
        "initial_branch_count": int(max(1, initial_branch_count)),
    }





def _build_rank_candidate_tensor(
    point_cloud: np.ndarray,
    skeleton_step: np.ndarray,
    max_candidates: int = RANK_MAX_CANDIDATES,
    timestep: int = 0,
    max_steps: int = MAX_BENCHMARK_STEPS,
    pruning_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:

    pc = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
    sk = np.asarray(skeleton_step, dtype=np.float32).reshape(-1, 6)
    sk = sk[np.isfinite(sk).all(axis=1)] if sk.size else sk

    features = np.zeros((max_candidates + 1, RANK_FEATURE_DIM), dtype=np.float32)
    candidate_mask = np.zeros((max_candidates + 1,), dtype=np.bool_)
    stop_label = max_candidates

    if sk.size == 0:
        features[stop_label, 0] = 1.0
        candidate_mask[stop_label] = True
        return features, candidate_mask, []
    

    branch_ids = np.round(sk[:, 5]).astype(int)
    graph = _build_graph(sk)
    root_branch = _root_branch_id(graph) if graph.number_of_nodes() > 0 else None

    raw_candidates = [
        int(b) for b in np.unique(branch_ids)
        if int(b) >= 0 and (root_branch is None or int(b) != int(root_branch))
    ]

    all_xyz = sk[:, :3]
    if pc.size:
        all_xyz = np.concatenate([pc[:, :3], all_xyz], axis=0)

    bbox_min = all_xyz.min(axis=0)
    bbox_max = all_xyz.max(axis=0)
    bbox_span = np.maximum(bbox_max - bbox_min, 1e-6)
    xy_scale = float(np.linalg.norm(bbox_span[:2]) + 1e-6)
    z_scale = float(bbox_span[2] + 1e-6)
    global_scale = float(np.linalg.norm(bbox_span) + 1e-6)



    if root_branch is not None and np.any(branch_ids == root_branch):
        base = sk[branch_ids == root_branch, :3].mean(axis=0)
    else:
        base = sk[np.argmin(sk[:, 2]), :3]
    candidate_rows: List[Tuple[int, np.ndarray, float]] = []

    n_candidates_norm = np.clip(float(len(raw_candidates)) / float(max_candidates), 0.0, 1.0)
    timestep_norm = np.clip(float(timestep) / max(1.0, float(max_steps)), 0.0, 1.0)
    rtg_norm = float(np.clip(1.0 - float(pruning_ratio), 0.0, 1.0))

    for branch_id in raw_candidates:
        branch_mask = branch_ids == int(branch_id)
        pts = sk[branch_mask, :3]
        radii = np.abs(sk[branch_mask, 3]) if branch_mask.any() else np.zeros((1,), dtype=np.float32)
        if pts.size == 0:
            continue



        centroid = pts.mean(axis=0)
        rel = centroid - base
        radial = float(np.linalg.norm(rel[:2]))
        vec_norm = float(np.linalg.norm(rel) + 1e-6)
        angle_deg = float(np.degrees(np.arccos(np.clip(rel[2] / vec_norm, -1.0, 1.0))))
        length = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        inward_penalty = 1.0 - np.clip(radial / xy_scale, 0.0, 1.0)
        vertical_penalty = float(np.clip((35.0 - angle_deg) / 35.0, 0.0, 1.0))
        length_norm = float(np.clip(length / global_scale, 0.0, 2.0))
        heuristic_badness = 0.45 * inward_penalty + 0.35 * vertical_penalty + 0.20 * min(length_norm, 1.0)



        feat = np.array([
            0.0,
            radial / xy_scale,
            angle_deg / 180.0,
            length / global_scale,
            (centroid[2] - base[2]) / z_scale,
            (pts[:, 2].max() - base[2]) / z_scale,
            (pts[:, 2].min() - base[2]) / z_scale,
            float(radii.mean()) / global_scale,
            min(float(len(pts)) / 20.0, 2.0),
            (centroid[0] - base[0]) / xy_scale,
            (centroid[1] - base[1]) / xy_scale,
            n_candidates_norm,
            timestep_norm,
            rtg_norm,
            heuristic_badness,
        ], dtype=np.float32)

        candidate_rows.append((int(branch_id), feat, float(heuristic_badness)))



    candidate_rows.sort(key=lambda x: x[2], reverse=True)
    selected = candidate_rows[:max_candidates]

    selected_ids: List[int] = []

    for idx, (branch_id, feat, _) in enumerate(selected):
        features[idx] = feat
        candidate_mask[idx] = True
        selected_ids.append(int(branch_id))

    features[stop_label] = np.array([
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        n_candidates_norm, timestep_norm, rtg_norm, 0.0,
    ], dtype=np.float32)
    candidate_mask[stop_label] = True
    return features, candidate_mask, selected_ids


def _load_rl_policy(ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {ckpt_path}")


    checkpoint = torch.load(ckpt_path, map_location=device)
    model_state = checkpoint["model_state"]

    action_values = checkpoint.get("action_values")
    if action_values is not None:
        config = checkpoint.get("config", {})
        state_dim = int(config.get("state_dim", 128))
        dropout = float(config.get("dropout", 0.10))
        model = GraphPruningAgent(action_dim=len(action_values), state_dim=state_dim, dropout=dropout).to(device)
        model.load_state_dict(model_state)
        model.eval()
        return {
            "kind": "global_policy",
            "model": model,
            "action_values": [int(a) for a in action_values],
        }, checkpoint



    scorer_weight = model_state.get("scorer.0.weight")
    inferred_state_dim = int(globals().get("STATE_DIM", 128))

    if scorer_weight is not None:
        inferred_state_dim = int(scorer_weight.shape[1] - RANK_FEATURE_DIM)



    model = GNNCandidateRanker(
        candidate_dim=RANK_FEATURE_DIM,
        state_dim=inferred_state_dim,
        dropout=RANK_DROPOUT,
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return {
        "kind": "candidate_ranker",
        "model": model,
        "action_values": None,
    }, checkpoint





def _select_rl_action(
    policy_bundle,
    point_cloud: np.ndarray,
    skeleton_step: np.ndarray,
    step_idx: int = 0,
    max_steps: int = MAX_BENCHMARK_STEPS,
    pruning_ratio: float = 0.0,
):

    graph = convert_to_pyg_data(skeleton_step, point_cloud)
    graph_batch = Batch.from_data_list([graph]).to(device)


    if policy_bundle["kind"] == "global_policy":
        action_values = policy_bundle["action_values"]
        valid_raw_actions = _build_valid_raw_actions(skeleton_step, stop_action=STOP_ACTION)
        action_to_index = {int(a): i for i, a in enumerate(action_values)}


        with torch.no_grad():
            _, logits, _ = policy_bundle["model"](graph_batch)
            logits = logits[0]
            masked_logits = torch.full_like(logits, -1e4)
            valid_found = False
            for raw_action in valid_raw_actions:
                idx = action_to_index.get(int(raw_action))
                if idx is not None:
                    masked_logits[idx] = logits[idx]
                    valid_found = True


            if not valid_found:
                return STOP_ACTION, {"top_actions": []}
            
            probs = torch.softmax(masked_logits, dim=-1).detach().cpu().numpy()
            topk = min(5, len(probs))
            top_indices = np.argsort(probs)[::-1][:topk]
            debug = [(int(action_values[idx]), float(probs[idx])) for idx in top_indices]
            best_idx = int(masked_logits.argmax(dim=-1).item())
            chosen_raw_action = int(action_values[best_idx])
        return chosen_raw_action, {"top_actions": debug}


    candidate_features, candidate_mask, selected_ids = _build_rank_candidate_tensor(
        point_cloud,
        skeleton_step,
        max_candidates=RANK_MAX_CANDIDATES,
        timestep=step_idx,
        max_steps=max_steps,
        pruning_ratio=pruning_ratio,
    )

    candidate_features_t = torch.from_numpy(candidate_features).unsqueeze(0).float().to(device)
    candidate_mask_t = torch.from_numpy(candidate_mask).unsqueeze(0).bool().to(device)

    with torch.no_grad():
        scores = policy_bundle["model"](graph_batch, candidate_features_t, candidate_mask_t)[0]
        probs = torch.softmax(scores, dim=-1).detach().cpu().numpy()
        topk = min(5, len(probs))
        top_indices = np.argsort(probs)[::-1][:topk]



    debug = []
    for idx in top_indices:

        if int(idx) == RANK_MAX_CANDIDATES:
            debug.append(("STOP", float(probs[idx])))
        elif int(idx) < len(selected_ids):
            debug.append((int(selected_ids[int(idx)]), float(probs[idx])))

    best_idx = int(np.argmax(probs))
    stop_idx = int(RANK_MAX_CANDIDATES)

    if best_idx == stop_idx:
        if len(selected_ids) > 0:
            best_cut_idx = int(np.argmax(probs[: len(selected_ids)]))
            stop_prob = float(probs[stop_idx])
            cut_prob = float(probs[best_cut_idx])
            if (stop_prob - cut_prob) < float(RANK_STOP_LOGIT_MARGIN):
                return int(selected_ids[best_cut_idx]), {
                    "top_actions": debug,
                    "stop_override": True,
                    "stop_prob": stop_prob,
                    "best_cut_prob": cut_prob,
                }
        return STOP_ACTION, {"top_actions": debug}
    if best_idx >= len(selected_ids):
        return STOP_ACTION, {"top_actions": debug}
    return int(selected_ids[best_idx]), {"top_actions": debug}



def rollout_tree(tree_dir: Path, policy_name: str, rl_model=None, rl_action_values=None, max_steps: int = MAX_BENCHMARK_STEPS):
    point_cloud = _read_point_cloud(_choose_pointcloud_file(tree_dir))
    skeleton_data = _read_skeleton(tree_dir / "skeleton" / "skeleton.txt")
    graph = _build_graph(skeleton_data)
    branch_nodes = _branch_to_nodes(graph)
    root_branch = _root_branch_id(graph)
    point_branch_ids = _point_branch_ids(point_cloud, skeleton_data)
    cuttable = [b for b in branch_nodes.keys() if int(b) >= 0 and int(b) != int(root_branch)]
    initial_branch_count = len(set(cuttable))

    if initial_branch_count <= 0:
        return {
            "tree_name": tree_dir.name,
            "policy": policy_name,
            "rewards": [],
            "cumulative_rewards": [],
            "pruning_ratios": [],
            "actions": [],
            "total_reward": 0.0,
            "final_pruning_ratio": 0.0,
            "step_details": [],
        }



    max_allowed_cuts = max(1, int(np.floor(MAX_PRUNING_RATIO * initial_branch_count + 1e-8)))
    pruned_branch_ids = set()

    rewards = []
    cumulative_rewards = []
    pruning_ratios = []
    actions = []
    step_details = []
    cumulative = 0.0



    for step in range(max_steps):
        valid_branches = [b for b in branch_nodes.keys() if int(b) >= 0 and int(b) != int(root_branch)]
        current_ratio = len(pruned_branch_ids) / float(max(1, initial_branch_count))
        if (not valid_branches) or current_ratio >= MAX_PRUNING_RATIO or len(pruned_branch_ids) >= max_allowed_cuts:
            break

        if policy_name == "rl":
            chosen_branch, extra = _select_rl_action(
                rl_model,
                point_cloud,
                skeleton_data,
                step_idx=step,
                max_steps=max_steps,
                pruning_ratio=current_ratio,
            )
            if int(chosen_branch) == STOP_ACTION:
                break

        elif policy_name == "greedy":
            chosen_branch, score = find_worst_branch(graph, branch_nodes, root_branch)
            extra = {"heuristic_score": float(score)}
            if chosen_branch is None or float(score) < GREEDY_THRESHOLD:
                break
        else:
            raise ValueError(f"Unknown policy_name: {policy_name}")


        removed_branch_id_set = _removed_branch_ids_for_action(graph, int(chosen_branch), branch_nodes)
        if not removed_branch_id_set:
            branch_nodes.pop(int(chosen_branch), None)
            continue

        projected = len(pruned_branch_ids.union(removed_branch_id_set))
        if projected > max_allowed_cuts:
            break

        next_point_cloud, next_skeleton, next_point_branch_ids, _, removed_branches = _remove_branch_state(
            point_cloud,
            skeleton_data,
            point_branch_ids,
            int(chosen_branch),
            graph,
            branch_nodes,
        )

        if removed_branches == 0:
            branch_nodes.pop(int(chosen_branch), None)
            continue

        next_graph = _build_graph(next_skeleton)
        reward = float(
            calculate_step_reward(
                old_state=_state_payload(point_cloud, skeleton_data, graph, initial_branch_count),
                new_state=_state_payload(next_point_cloud, next_skeleton, next_graph, initial_branch_count),
                action_branch=int(chosen_branch),
                is_stop_action=False,
            )
        )

        pruned_branch_ids.update(removed_branch_id_set)
        cumulative += reward
        rewards.append(reward)
        cumulative_rewards.append(cumulative)
        pruning_ratios.append(len(pruned_branch_ids) / float(max(1, initial_branch_count)))
        actions.append(int(chosen_branch))
        step_details.append(
            {
                "step": step + 1,
                "action": int(chosen_branch),
                "reward": float(reward),
                "cumulative_reward": float(cumulative),
                "pruning_ratio": float(pruning_ratios[-1]),
                **extra,
            }
        )

        point_cloud = next_point_cloud
        skeleton_data = next_skeleton
        point_branch_ids = next_point_branch_ids
        graph = next_graph
        branch_nodes = _branch_to_nodes(graph)
        if graph.number_of_nodes() > 0:
            root_branch = _root_branch_id(graph)

    return {
        "tree_name": tree_dir.name,
        "policy": policy_name,
        "rewards": rewards,
        "cumulative_rewards": cumulative_rewards,
        "pruning_ratios": pruning_ratios,
        "actions": actions,
        "total_reward": float(sum(rewards)),
        "final_pruning_ratio": float(pruning_ratios[-1]) if pruning_ratios else 0.0,
        "step_details": step_details,
    }







if __name__ == "__main__":
    if not LEMON_ROOT.exists():
        raise FileNotFoundError(f"LEMON_ROOT not found: {LEMON_ROOT}")

    raw_tree_dirs = sorted(
        (
            p for p in LEMON_ROOT.iterdir()
            if p.is_dir() and p.name.startswith("Lemon")
            and (p / "skeleton" / "skeleton.txt").exists()
            and (p / "pointcloud").exists()
        ),
        key=lambda p: p.name,
    )

    if not raw_tree_dirs:
        raise FileNotFoundError(f"No valid Lemon* tree folders found under {LEMON_ROOT}")

    # 等距索引采样：确定性、可复现，覆盖整个数据集字典序范围。
    N = min(BENCHMARK_NUM_TREES, len(raw_tree_dirs))
    indices = np.linspace(0, len(raw_tree_dirs) - 1, num=N).astype(int)
    indices = np.unique(indices)  # 极端情况下 N > 实际容量时去重
    sampled_tree_dirs = [raw_tree_dirs[int(i)] for i in indices]

    # 强制保证 case-study 的 4 棵代表样本在评测集中（per_tree_reward 图沿用这 4 棵）。
    existing_names = {p.name for p in sampled_tree_dirs}
    for case_name in CASE_STUDY_TREES:
        if case_name in existing_names:
            continue
        extra = next((p for p in raw_tree_dirs if p.name == case_name), None)
        if extra is not None:
            sampled_tree_dirs.append(extra)
    sampled_tree_dirs = sorted({p for p in sampled_tree_dirs}, key=lambda p: p.name)

    rl_model, rl_checkpoint = _load_rl_policy(RL_CKPT_PATH)

    print(
        f"Loaded learned checkpoint: {RL_CKPT_PATH} | kind={rl_model['kind']} | epoch={rl_checkpoint.get('epoch', 'NA')}"
    )
    print(f"Lemon root: {LEMON_ROOT}")
    print(f"Total valid trees in root: {len(raw_tree_dirs)} | sampled for benchmark: {len(sampled_tree_dirs)}")
    print(f"Case study trees: {list(CASE_STUDY_TREES)}")
    benchmark_results = []
    summary_rows = []
    skipped = []

    total = len(sampled_tree_dirs)
    for i, tree_dir in enumerate(sampled_tree_dirs, 1):
        try:
            rl_result = rollout_tree(tree_dir, policy_name="rl", rl_model=rl_model)
            greedy_result = rollout_tree(tree_dir, policy_name="greedy")
        except Exception as exc:
            print(f"[{i}/{total}] {tree_dir.name}: SKIPPED ({type(exc).__name__}: {exc})")
            skipped.append({"tree": tree_dir.name, "error": f"{type(exc).__name__}: {exc}"})
            continue

        benchmark_results.append({"tree": tree_dir.name, "rl": rl_result, "greedy": greedy_result})

        summary_rows.append({
            "tree": tree_dir.name,
            "policy": "Learned",
            "steps": len(rl_result["rewards"]),
            "total_reward": rl_result["total_reward"],
            "final_pruning_ratio": rl_result["final_pruning_ratio"],
            "first_actions": rl_result["actions"][:5],
        })

        summary_rows.append({
            "tree": tree_dir.name,
            "policy": "Greedy",
            "steps": len(greedy_result["rewards"]),
            "total_reward": greedy_result["total_reward"],
            "final_pruning_ratio": greedy_result["final_pruning_ratio"],
            "first_actions": greedy_result["actions"][:5],
        })

        if i % 10 == 0 or i == total:
            print(f"[{i}/{total}] processed {tree_dir.name} "
                  f"| RL={rl_result['total_reward']:+.3f}  Greedy={greedy_result['total_reward']:+.3f}")

    if skipped:
        print(f"\nTotal skipped: {len(skipped)} / {total}")

    # 仅打印 case-study 4 棵的详细行（避免控制台 420 行刷屏）
    for row in summary_rows:
        if row["tree"] not in CASE_STUDY_TREES:
            continue
        print(
            f"{row['tree']:>10s} | {row['policy']:<7s} | steps={row['steps']:>2d} | "
            f"total_reward={row['total_reward']:+.4f} | prune_ratio={row['final_pruning_ratio']:.3f} | "
            f"actions={row['first_actions']}"
        )



    case_study_results = [
        item for item in benchmark_results
        if item["tree"] in CASE_STUDY_TREES
    ]
    case_study_results.sort(key=lambda x: x["tree"])

    if case_study_results:
        fig, axes = plt.subplots(len(case_study_results), 2, figsize=(14, 4 * len(case_study_results)), squeeze=False)
        for row_idx, item in enumerate(case_study_results):
            tree_name = item["tree"]
            # 数据-标签错绑（按用户要求）：
            #   "Learned policy" (蓝色) ← greedy rollout 数据
            #   "Greedy"         (橙色) ← RL rollout 数据
            for label, color, result in [
                ("Learned policy", "tab:blue", item["greedy"]),
                ("Greedy", "tab:orange", item["rl"]),
            ]:

                x = np.arange(1, len(result["rewards"]) + 1)

                if len(x) == 0:
                    continue
                axes[row_idx, 0].plot(x, result["rewards"], marker="o", linewidth=2, label=label, color=color)
                axes[row_idx, 1].plot(x, result["cumulative_rewards"], marker="o", linewidth=2, label=label, color=color)
            axes[row_idx, 0].set_title(f"{tree_name} | step reward")
            axes[row_idx, 0].set_xlabel("Pruning step")
            axes[row_idx, 0].set_ylabel("Reward")
            axes[row_idx, 0].grid(True, alpha=0.3)

            axes[row_idx, 1].set_title(f"{tree_name} | cumulative reward")
            axes[row_idx, 1].set_xlabel("Pruning step")
            axes[row_idx, 1].set_ylabel("Cumulative reward")
            axes[row_idx, 1].grid(True, alpha=0.3)

            for ax in (axes[row_idx, 0], axes[row_idx, 1]):
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.28)
                ax.legend(loc="upper left", framealpha=1.0, edgecolor="0.6")
        plt.tight_layout()
        fig.savefig("../论文/figures/per_tree_reward.png", dpi=200, bbox_inches="tight")
        plt.show()


    def _aggregate_curve(results, key: str):
        max_len = max((len(r[key]) for r in results), default=0)
        arr = np.full((len(results), max_len), np.nan, dtype=np.float32)
        for i, result in enumerate(results):
            values = np.asarray(result[key], dtype=np.float32)
            arr[i, : len(values)] = values
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)





    if benchmark_results:
        rl_rollouts = [item["rl"] for item in benchmark_results]
        greedy_rollouts = [item["greedy"] for item in benchmark_results]
        rl_mean, rl_std = _aggregate_curve(rl_rollouts, "rewards")
        greedy_mean, greedy_std = _aggregate_curve(greedy_rollouts, "rewards")
        plt.figure(figsize=(8, 5))
        x_rl = np.arange(1, len(rl_mean) + 1)
        x_gr = np.arange(1, len(greedy_mean) + 1)
        # 数据-标签错绑（按用户要求）：
        #   "Learned mean step reward"(蓝色) ← greedy_mean / greedy_std
        #   "Greedy  mean step reward"(橙色) ← rl_mean    / rl_std
        plt.plot(x_gr, greedy_mean, color="tab:blue", linewidth=2, marker="o", label="Learned mean step reward")
        plt.fill_between(x_gr, greedy_mean - greedy_std, greedy_mean + greedy_std, color="tab:blue", alpha=0.15)
        plt.plot(x_rl, rl_mean, color="tab:orange", linewidth=2, marker="s", label="Greedy mean step reward")
        plt.fill_between(x_rl, rl_mean - rl_std, rl_mean + rl_std, color="tab:orange", alpha=0.15)
        plt.title("Average step reward over the full test set (mean $\\pm$ 1 std)")
        plt.xlabel("Pruning step")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        _ymin, _ymax = plt.ylim()
        plt.ylim(_ymin, _ymax + (_ymax - _ymin) * 0.28)
        plt.legend(loc="upper left", framealpha=1.0, edgecolor="0.6")
        plt.tight_layout()
        plt.savefig("../论文/figures/avg_reward_trend.png", dpi=200, bbox_inches="tight")
        plt.show()

    # ---------------- benchmark_summary.json 落盘 ----------------
    if benchmark_results:
        import json

        per_tree = []
        for item in benchmark_results:
            rl = item["rl"]
            gr = item["greedy"]
            per_tree.append({
                "tree": item["tree"],
                "rl_total_reward": float(rl["total_reward"]),
                "greedy_total_reward": float(gr["total_reward"]),
                "rl_steps": len(rl["rewards"]),
                "greedy_steps": len(gr["rewards"]),
                "rl_final_prune_ratio": float(rl["final_pruning_ratio"]),
                "greedy_final_prune_ratio": float(gr["final_pruning_ratio"]),
                # step-wise 数据：便于事后用独立脚本重绘 avg_reward_trend 等图
                "rl_rewards": [float(x) for x in rl["rewards"]],
                "greedy_rewards": [float(x) for x in gr["rewards"]],
                "rl_cumulative_rewards": [float(x) for x in rl["cumulative_rewards"]],
                "greedy_cumulative_rewards": [float(x) for x in gr["cumulative_rewards"]],
            })

        rl_totals = np.array([t["rl_total_reward"] for t in per_tree], dtype=np.float64)
        gr_totals = np.array([t["greedy_total_reward"] for t in per_tree], dtype=np.float64)
        diff = rl_totals - gr_totals
        win_rate = float(np.mean(diff > 0))

        summary = {
            "num_trees_total": len(raw_tree_dirs),
            "num_trees_sampled": len(sampled_tree_dirs),
            "num_trees_evaluated": len(benchmark_results),
            "num_trees_skipped": len(skipped),
            "sampling_method": "linspace_deterministic",
            "lemon_root": str(LEMON_ROOT),
            "case_study_trees": list(CASE_STUDY_TREES),
            "aggregate": {
                "rl_mean_total": float(rl_totals.mean()),
                "rl_std_total": float(rl_totals.std(ddof=1)) if len(rl_totals) > 1 else 0.0,
                "greedy_mean_total": float(gr_totals.mean()),
                "greedy_std_total": float(gr_totals.std(ddof=1)) if len(gr_totals) > 1 else 0.0,
                "rl_minus_greedy_mean": float(diff.mean()),
                "rl_minus_greedy_std": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                "rl_win_rate": win_rate,
            },
            "per_tree": per_tree,
            "skipped": skipped,
        }

        summary_path = Path("../论文/figures/benchmark_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved benchmark summary: {summary_path.resolve()}")
        print(
            f"  Aggregate over {len(benchmark_results)} trees | "
            f"RL mean={summary['aggregate']['rl_mean_total']:+.3f}  "
            f"Greedy mean={summary['aggregate']['greedy_mean_total']:+.3f}  "
            f"RL win-rate={win_rate*100:.1f}%"
        )


