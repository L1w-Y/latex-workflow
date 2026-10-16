from __future__ import annotations

"""
GNN-first RL core for fruit-tree pruning.

本模块提供三个核心组件：
1) `GATStateEncoder`：使用 PyG 的 GAT 对骨架拓扑做图编码，并融合轻量 PointNet 点云分支。
2) `GNNSurrogateEnv`：离线 surrogate/world-model，用于快速预测动作带来的 6 个 KPI 增量。
3) `TreeToPyGGraph` / `convert_to_pyg_data`：将原始 skeleton / point cloud 数组稳健地转换为 PyG `Data`。

设计上吸取了此前训练中的几个教训：
- **避免 CPU 瓶颈**：拓扑直接转为 `edge_index`，供 PyG 在 GPU 路径中批处理。
- **避免对原始 branch_id 过拟合**：虽然保留 `branch_id` 这一维，但会先做秩次归一化，
  而不是直接把全局编号原样喂给网络，减少“记编号”而不是“学结构”的风险。
- **严格支持 PyG batching**：图池化依赖 `data.batch`，点云池化优先使用 `point_cloud_batch`，
  没有时回退到 `num_points` 自动恢复 batch 索引。
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

PyGDataLike = Union[Data, Batch]


# ============================================================
# 1) Data transformation utilities
# ============================================================
class TreeToPyGGraph:
    """
    将标准 numpy tree state 转成 PyG `Data`。

    输入约定：
    - `skeleton_array`: [num_nodes, 6]
        列顺序为 [x, y, z, r, parent_idx, branch_id]
    - `point_cloud_array`: [num_points, 3]

    输出字段：
    - `x`: [num_nodes, 6] 的节点特征（已做数值归一化）
    - `edge_index`: [2, num_edges]，方向为 parent -> child
    - `point_cloud`: [num_points, 3]，与骨架对齐后的归一化点云
    - `pos`: [num_nodes, 3]，归一化 xyz，可供后续可视化或 edge/pos-based GNN 扩展使用
    - `num_points`: [1]，用于 batch 后恢复 point_cloud 的 graph assignment
    - `root_branch`: [1]，估计出来的主干/根分支 ID
    - `tree_scale`: [1]，几何归一化尺度
    """

    @staticmethod
    def _sanitize_skeleton(skeleton_array: np.ndarray) -> np.ndarray:
        sk = np.asarray(skeleton_array, dtype=np.float32)
        if sk.ndim != 2:
            raise ValueError(f"skeleton_array must be 2D, got shape {sk.shape}")

        if sk.size == 0:
            return np.zeros((0, 6), dtype=np.float32)

        if sk.shape[1] < 6:
            raise ValueError(
                f"skeleton_array must have at least 6 columns [x,y,z,r,parent_idx,branch_id], got {sk.shape[1]}"
            )

        sk = sk[:, :6]
        sk = np.nan_to_num(sk, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return sk

    @staticmethod
    def _sanitize_point_cloud(point_cloud_array: np.ndarray) -> np.ndarray:
        pc = np.asarray(point_cloud_array, dtype=np.float32)
        if pc.ndim != 2:
            raise ValueError(f"point_cloud_array must be 2D, got shape {pc.shape}")

        if pc.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if pc.shape[1] < 3:
            raise ValueError(f"point_cloud_array must have at least 3 columns [x,y,z], got {pc.shape[1]}")

        pc = pc[:, :3]
        pc = np.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return pc

    @staticmethod
    def _infer_root_branch_id(skeleton_array: np.ndarray) -> int:
        if skeleton_array.size == 0:
            return 0

        parent_idx = np.round(skeleton_array[:, 4]).astype(np.int64)
        branch_ids = np.round(skeleton_array[:, 5]).astype(np.int64)
        root_mask = parent_idx <= 0

        if root_mask.any():
            vals, counts = np.unique(branch_ids[root_mask], return_counts=True)
            return int(vals[np.argmax(counts)])

        valid_branch_ids = [int(b) for b in np.unique(branch_ids) if int(b) >= 0]
        if not valid_branch_ids:
            return 0
        return int(valid_branch_ids[0])

    @staticmethod
    def _infer_parent_mode(parent_raw: np.ndarray, num_nodes: int) -> str:
        """
        自动判断 parent_idx 更像 1-based 还是 0-based。
        为了稳健性，分别计算两种解释下的“有效边数量”，选择更合理的一种。
        """
        if num_nodes <= 1:
            return "one_based"

        child_ids = np.arange(num_nodes, dtype=np.int64)
        parent_raw = np.round(parent_raw).astype(np.int64)

        parent_one_based = parent_raw - 1
        valid_one = (parent_raw > 0) & (parent_one_based >= 0) & (parent_one_based < num_nodes) & (parent_one_based != child_ids)
        score_one = int(valid_one.sum())

        parent_zero_based = parent_raw
        valid_zero = (parent_raw >= 0) & (parent_zero_based < num_nodes) & (parent_zero_based != child_ids)
        score_zero = int(valid_zero.sum())

        return "one_based" if score_one >= score_zero else "zero_based"

    @staticmethod
    def build_edge_index(skeleton_array: np.ndarray) -> Tensor:
        """
        从 skeleton 的 `parent_idx` 动态构造有向边：parent -> child。

        对 disconnected / root / 非法 parent 都会自动跳过，不会报错。
        """
        sk = TreeToPyGGraph._sanitize_skeleton(skeleton_array)
        num_nodes = sk.shape[0]

        if num_nodes == 0:
            return torch.empty((2, 0), dtype=torch.long)

        parent_raw = sk[:, 4].astype(np.int64)
        mode = TreeToPyGGraph._infer_parent_mode(parent_raw, num_nodes)

        senders = []
        receivers = []
        for child_idx in range(num_nodes):
            p = int(parent_raw[child_idx])
            if mode == "one_based":
                parent_idx = p - 1
                is_root = p <= 0
            else:
                parent_idx = p
                is_root = p < 0 or p == child_idx

            if is_root:
                continue
            if parent_idx < 0 or parent_idx >= num_nodes:
                continue
            if parent_idx == child_idx:
                continue

            senders.append(parent_idx)
            receivers.append(child_idx)

        if not senders:
            return torch.empty((2, 0), dtype=torch.long)

        edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        return edge_index.contiguous()

    @staticmethod
    def _normalize_geometry(
        skeleton_array: np.ndarray,
        point_cloud_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        """
        对骨架和点云做统一坐标归一化。

        归一化策略：
        - 以 root/base 附近的骨架中心为参考原点
        - 以整棵树的包围盒尺度做缩放

        这样做的目的，是避免模型死记绝对世界坐标，提高跨树泛化能力。
        """
        sk = TreeToPyGGraph._sanitize_skeleton(skeleton_array)
        pc = TreeToPyGGraph._sanitize_point_cloud(point_cloud_array)

        if sk.shape[0] == 0:
            base = np.zeros((3,), dtype=np.float32)
            if pc.shape[0] > 0:
                scale = float(np.linalg.norm(pc.max(axis=0) - pc.min(axis=0)) + 1e-6)
                pc_norm = (pc - pc.mean(axis=0, keepdims=True)) / scale
            else:
                scale = 1.0
                pc_norm = pc
            return sk, pc_norm, base, scale, 0

        parent_idx = np.round(sk[:, 4]).astype(np.int64)
        root_mask = parent_idx <= 0
        root_branch = TreeToPyGGraph._infer_root_branch_id(sk)

        if root_mask.any():
            base = sk[root_mask, :3].mean(axis=0)
        else:
            base = sk[np.argmin(sk[:, 2]), :3]

        all_xyz = sk[:, :3]
        if pc.shape[0] > 0:
            all_xyz = np.concatenate([all_xyz, pc[:, :3]], axis=0)

        scale = float(np.linalg.norm(all_xyz.max(axis=0) - all_xyz.min(axis=0)) + 1e-6)
        sk_norm = sk.copy()
        sk_norm[:, :3] = (sk_norm[:, :3] - base[None, :]) / scale

        if pc.shape[0] > 0:
            pc_norm = (pc - base[None, :]) / scale
        else:
            pc_norm = pc

        return sk_norm, pc_norm, base.astype(np.float32), scale, int(root_branch)

    @staticmethod
    def convert_to_pyg_data(
        skeleton_array: np.ndarray,
        point_cloud_array: np.ndarray,
    ) -> Data:
        """
        主转换函数：返回可直接送入 PyG 模型的 `Data`。

        注意：
        - 保留了 6 维节点特征的接口语义：`[x, y, z, r, parent_idx, branch_id]`
        - 但其中的 `parent_idx` / `branch_id` 已做归一化编码，避免原始编号导致的 shortcut learning
        """
        sk = TreeToPyGGraph._sanitize_skeleton(skeleton_array)
        pc = TreeToPyGGraph._sanitize_point_cloud(point_cloud_array)

        if sk.shape[0] == 0:
            # 极端情况下用一个 dummy node 保持 GNN 前向稳定
            x = torch.zeros((1, 6), dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            point_cloud = torch.from_numpy(pc).float()
            data = Data(x=x, edge_index=edge_index, point_cloud=point_cloud)
            data.pos = torch.zeros((1, 3), dtype=torch.float32)
            data.num_points = torch.tensor([point_cloud.size(0)], dtype=torch.long)
            data.root_branch = torch.tensor([0], dtype=torch.long)
            data.tree_scale = torch.tensor([1.0], dtype=torch.float32)
            return data

        sk_norm, pc_norm, _, tree_scale, root_branch = TreeToPyGGraph._normalize_geometry(sk, pc)
        edge_index = TreeToPyGGraph.build_edge_index(sk)

        radius = np.abs(sk_norm[:, 3:4])
        radius = radius / (radius.max() + 1e-6)

        parent_raw = np.round(sk[:, 4]).astype(np.int64)
        parent_feat = np.clip(parent_raw.astype(np.float32), a_min=0.0, a_max=None)
        parent_feat = parent_feat / max(1.0, float(len(sk) - 1))
        parent_feat = parent_feat.reshape(-1, 1)

        # 吸取之前“branch_id 编号本身没有跨树语义”的教训：
        # 这里不直接使用原始 branch_id，而是映射为当前树内的 rank-normalized 值。
        branch_raw = np.round(sk[:, 5]).astype(np.int64)
        unique_branch = sorted(int(v) for v in np.unique(branch_raw))
        branch_rank = {bid: idx for idx, bid in enumerate(unique_branch)}
        branch_feat = np.array([branch_rank[int(b)] for b in branch_raw], dtype=np.float32).reshape(-1, 1)
        branch_feat = branch_feat / max(1.0, float(len(unique_branch) - 1))

        node_features = np.concatenate(
            [
                sk_norm[:, :3],
                radius,
                parent_feat,
                branch_feat,
            ],
            axis=1,
        ).astype(np.float32)

        data = Data(
            x=torch.from_numpy(node_features).float(),
            edge_index=edge_index,
            point_cloud=torch.from_numpy(pc_norm).float(),
        )
        data.pos = torch.from_numpy(sk_norm[:, :3]).float()
        data.num_points = torch.tensor([pc_norm.shape[0]], dtype=torch.long)
        data.root_branch = torch.tensor([root_branch], dtype=torch.long)
        data.tree_scale = torch.tensor([tree_scale], dtype=torch.float32)
        return data


def convert_to_pyg_data(skeleton_array: np.ndarray, point_cloud_array: np.ndarray) -> Data:
    """便捷封装，直接暴露为函数接口。"""
    return TreeToPyGGraph.convert_to_pyg_data(skeleton_array, point_cloud_array)


# ============================================================
# 2) GNN State Encoder
# ============================================================
class _PointNetLite(nn.Module):
    """轻量 PointNet 风格点云编码器，输出固定长度 64 维 embedding。"""

    def __init__(self, hidden_dim: int = 128, out_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        point_cloud: Tensor,
        point_batch: Optional[Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> Tensor:
        if point_cloud.ndim == 3:
            # Dense path: [B, N, 3]
            feat = self.point_mlp(point_cloud)
            return feat.max(dim=1).values

        if point_cloud.ndim != 2 or point_cloud.size(-1) != 3:
            raise ValueError(f"point_cloud must be [B, N, 3] or [num_points, 3], got {tuple(point_cloud.shape)}")

        if point_batch is None:
            # 单图场景默认所有点属于同一 graph
            point_batch = torch.zeros(point_cloud.size(0), dtype=torch.long, device=point_cloud.device)

        feat = self.point_mlp(point_cloud)
        pooled = global_max_pool(feat, point_batch, size=num_graphs)
        return pooled


class GATStateEncoder(nn.Module):
    """
    GNN 版状态编码器。

    输入：
    - `data.x`: [num_nodes, 6]，节点特征
    - `data.edge_index`: [2, num_edges]，树拓扑
    - `data.point_cloud`: [num_points, 3] 或 [B, N, 3]，归一化点云

    输出：
    - `[num_graphs, 128]` 的 state vector
    """

    def __init__(
        self,
        node_in_dim: int = 6,
        hidden_dim: int = 128,
        graph_out_dim: int = 64,
        point_out_dim: int = 64,
        state_dim: int = 128,
        num_gat_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        pool: str = "meanmax",
    ) -> None:
        super().__init__()

        if num_gat_layers < 2:
            raise ValueError("num_gat_layers must be >= 2")
        if pool not in {"mean", "max", "meanmax"}:
            raise ValueError("pool must be one of {'mean', 'max', 'meanmax'}")

        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.graph_out_dim = graph_out_dim
        self.point_out_dim = point_out_dim
        self.state_dim = state_dim
        self.pool = pool

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        in_dim = node_in_dim
        for _ in range(num_gat_layers):
            conv = GATConv(
                in_channels=in_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=True,
            )
            self.gat_layers.append(conv)
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        pooled_dim = hidden_dim * 2 if pool == "meanmax" else hidden_dim
        self.graph_projector = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, graph_out_dim),
        )

        self.point_encoder = _PointNetLite(hidden_dim=hidden_dim, out_dim=point_out_dim, dropout=dropout)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(graph_out_dim + point_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )

    @staticmethod
    def _num_graphs_from_batch(batch_vec: Tensor) -> int:
        if batch_vec.numel() == 0:
            return 1
        return int(batch_vec.max().item()) + 1

    @staticmethod
    def _recover_point_batch(data: PyGDataLike, point_cloud: Tensor, num_graphs: int) -> Tensor:
        """
        优先使用 `point_cloud_batch`；若没有，则从 `num_points` 恢复每个点所属 graph。
        """
        if hasattr(data, "point_cloud_batch") and getattr(data, "point_cloud_batch") is not None:
            point_batch = getattr(data, "point_cloud_batch")
            return point_batch.to(point_cloud.device)

        if hasattr(data, "num_points") and getattr(data, "num_points") is not None:
            counts = getattr(data, "num_points")
            if not torch.is_tensor(counts):
                counts = torch.tensor(counts, dtype=torch.long, device=point_cloud.device)
            counts = counts.view(-1).long().to(point_cloud.device)
            if counts.numel() == 1 and num_graphs == 1:
                return torch.zeros(point_cloud.size(0), dtype=torch.long, device=point_cloud.device)

            point_batch = torch.repeat_interleave(
                torch.arange(counts.numel(), device=point_cloud.device),
                counts.clamp(min=0),
            )
            if point_batch.numel() != point_cloud.size(0):
                raise ValueError(
                    "Cannot recover point batch assignment: concatenated point count does not match `num_points`."
                )
            return point_batch

        if num_graphs == 1:
            return torch.zeros(point_cloud.size(0), dtype=torch.long, device=point_cloud.device)

        raise ValueError(
            "Batched point clouds require either `data.point_cloud_batch` or `data.num_points` to recover graph membership."
        )

    def _encode_graph(self, x: Tensor, edge_index: Tensor, batch_vec: Tensor) -> Tensor:
        h = x
        for conv, norm in zip(self.gat_layers, self.gat_norms):
            residual = h if h.size(-1) == self.hidden_dim else None
            h = conv(h, edge_index)
            h = norm(h)
            h = torch.relu(h)
            if residual is not None:
                h = h + residual

        if self.pool == "mean":
            pooled = global_mean_pool(h, batch_vec)
        elif self.pool == "max":
            pooled = global_max_pool(h, batch_vec)
        else:
            pooled = torch.cat(
                [global_mean_pool(h, batch_vec), global_max_pool(h, batch_vec)],
                dim=-1,
            )

        return self.graph_projector(pooled)

    def _encode_points(self, data: PyGDataLike, num_graphs: int) -> Tensor:
        if not hasattr(data, "point_cloud") or getattr(data, "point_cloud") is None:
            return torch.zeros((num_graphs, self.point_out_dim), device=data.x.device, dtype=data.x.dtype)

        point_cloud = getattr(data, "point_cloud")
        if not torch.is_tensor(point_cloud):
            point_cloud = torch.as_tensor(point_cloud, dtype=torch.float32, device=data.x.device)
        else:
            point_cloud = point_cloud.to(data.x.device, dtype=torch.float32)

        if point_cloud.ndim == 3:
            return self.point_encoder(point_cloud, point_batch=None, num_graphs=point_cloud.size(0))

        point_batch = self._recover_point_batch(data, point_cloud, num_graphs=num_graphs)
        return self.point_encoder(point_cloud, point_batch=point_batch, num_graphs=num_graphs)

    def forward(self, data: PyGDataLike) -> Tensor:
        if not hasattr(data, "x") or not hasattr(data, "edge_index"):
            raise ValueError("Input data must contain `x` and `edge_index`.")

        x = data.x
        edge_index = data.edge_index

        if not torch.is_tensor(x) or x.ndim != 2 or x.size(-1) != self.node_in_dim:
            raise ValueError(f"data.x must be [num_nodes, {self.node_in_dim}], got {tuple(x.shape)}")
        if not torch.is_tensor(edge_index) or edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(f"data.edge_index must be [2, num_edges], got {tuple(edge_index.shape)}")

        x = x.float()
        edge_index = edge_index.long()

        if hasattr(data, "batch") and getattr(data, "batch") is not None:
            batch_vec = getattr(data, "batch").to(x.device)
        else:
            batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = self._num_graphs_from_batch(batch_vec)

        graph_emb = self._encode_graph(x, edge_index, batch_vec)
        point_emb = self._encode_points(data, num_graphs=num_graphs)

        if point_emb.size(0) != graph_emb.size(0):
            raise ValueError(
                f"Point cloud embedding batch size ({point_emb.size(0)}) does not match graph embedding batch size ({graph_emb.size(0)})."
            )

        fused = torch.cat([graph_emb, point_emb], dim=-1)
        state = self.fusion_mlp(fused)
        return state


# ============================================================
# 3) Surrogate reward environment / world model
# ============================================================
class GNNSurrogateEnv(nn.Module):
    """
    基于 GNN state vector 的 surrogate world model。

    输入：
    - `state_vec`: [B, 128]
    - `action`:    [B]，离散动作 id（建议事先把 raw branch_id 映射到连续索引）

    输出：
    - [B, 6]，表示预测的 KPI 增量：
      [ΔH, ΔC, ΔB, ΔL, ΔP, ΔS]
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int = 128,
        action_emb_dim: int = 64,
        hidden_dims: Sequence[int] = (256, 128),
        output_dim: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if state_dim <= 0:
            raise ValueError("state_dim must be > 0")
        if output_dim != 6:
            raise ValueError("output_dim is expected to be 6 for [ΔH, ΔC, ΔB, ΔL, ΔP, ΔS]")

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_emb_dim = action_emb_dim
        self.output_dim = output_dim

        self.action_embed = nn.Embedding(action_dim, action_emb_dim)

        layers = []
        in_dim = state_dim + action_emb_dim  # 128 + 64 = 192
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.world_model = nn.Sequential(*layers)

    def forward(self, state_vec: Tensor, action: Tensor) -> Tensor:
        if state_vec.ndim != 2 or state_vec.size(-1) != self.state_dim:
            raise ValueError(f"state_vec must be [B, {self.state_dim}], got {tuple(state_vec.shape)}")

        if action.ndim == 2 and action.size(-1) == 1:
            action = action.squeeze(-1)
        if action.ndim != 1:
            raise ValueError(f"action must be [B] or [B, 1], got {tuple(action.shape)}")
        if action.size(0) != state_vec.size(0):
            raise ValueError("Batch size of `action` must match `state_vec`.")

        # 说明：这里假设动作已事先 remap 到 [0, action_dim-1]。
        # 若上游仍保留 raw branch_id，请先构建一个 action vocabulary 再送入该模块。
        safe_action = action.long().clamp(min=0, max=self.action_dim - 1)
        action_emb = self.action_embed(safe_action)

        fused = torch.cat([state_vec, action_emb], dim=-1)
        delta_kpis = self.world_model(fused)
        return delta_kpis


__all__ = [
    "TreeToPyGGraph",
    "convert_to_pyg_data",
    "GATStateEncoder",
    "GNNSurrogateEnv",
]
