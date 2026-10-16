"""骨架分支级图状态编码器：分支为节点 + 边感知图注意力 → 整树状态向量 + 逐分枝向量。

设计要点（与论文 sec:formalization:graph 的分支级图对齐）：
- 节点 = 分枝（branch_id），节点特征为分枝级属性（几何、相对全树均值的
  比例、相对冠层中心的位置、绝对尺寸、子树规模、近邻交叉线索），与
  reward.py 的 BranchProps 计算口径同源
- 两类边：父分枝↔子分枝（拓扑）；非父子分枝间骨架点最小距离
  ≤ SPATIAL_RADIUS_FACTOR·d_cross（空间邻近，交叉/重叠判别所需信息）
- 边特征 = [是否拓扑边, 是否空间边, 最小距离/d_cross, 方向余弦]，
  通过 TransformerConv 进入消息本身（而不仅是注意力权重）
- 输出：整树状态向量 [G, state_dim]（DT 的 state token）与逐分枝向量
  [G, N, hidden]（动作头对每根分枝打分），动作空间与具体树的分枝编号无关
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch_geometric.data import Batch, Data
from torch_geometric.nn import TransformerConv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from config import CFG
from tree_data import BRANCH, build_parent_map

BRANCH_FEAT_DIM = 20
EDGE_FEAT_DIM = 4
GLOBAL_FEAT_DIM = 2            # [当前累计剪枝比, 当前分枝数/1000]
SPATIAL_RADIUS_FACTOR = 2.0
MAX_SPATIAL_PER_NODE = 16


# ==============================================================================
# 图构建
# ==============================================================================

def graph_arrays(sk: np.ndarray, parent_map: dict[int, int] | None = None,
                 pruning_ratio: float = 0.0) -> dict:
    """骨架 [N,6] → 分支级图的 numpy 数组。

    返回 {"x": [M,F] float32, "edge_index": [2,E] int64,
          "edge_attr": [E,4] float32, "g": [2] float32,
          "bids": [M] int64, "root": int}
    节点按 branch_id 升序排列；root 为主干在节点序中的位置。
    """
    cfg = CFG.reward
    n = sk.shape[0]
    bids = sk[:, BRANCH].astype(np.int64)

    # 稳定排序使同一分支的点连续（保持分支内文件顺序，与 compute_branch_props 一致）
    order = np.argsort(bids, kind="stable")
    sk_s = sk[order]
    bid_s = bids[order]
    boundaries = np.flatnonzero(np.diff(bid_s)) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, n]
    m = len(starts)
    unique_bids = bid_s[starts]
    counts = ends - starts
    seg_id = np.repeat(np.arange(m), counts)

    xyz = sk_s[:, :3].astype(np.float64)
    rad = sk_s[:, 3].astype(np.float64)

    origin = xyz[np.argmin(xyz[:, 2])]
    scale = float(np.max(xyz.max(axis=0) - xyz.min(axis=0))) or 1.0
    canopy = xyz.mean(axis=0)

    center = np.add.reduceat(xyz, starts, axis=0) / counts[:, None]
    vec = xyz[ends - 1] - xyz[starts]
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6)
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    same = bid_s[:-1] == bid_s[1:]
    length = np.bincount(seg_id[:-1][same], weights=d[same], minlength=m)
    radius = np.bincount(seg_id, weights=rad, minlength=m) / counts
    volume = length * np.pi * radius ** 2

    eff = length > cfg.l_eff
    mean_len = float(length[eff].mean()) if eff.any() else float(length.mean()) or 1.0
    mean_r = float(radius[eff].mean()) if eff.any() else float(radius.mean()) or 1.0

    # 父子关系：全平台唯一实现 tree_data.build_parent_map
    pmap = parent_map if parent_map is not None else build_parent_map(sk)
    parent_bid = np.array([pmap.get(int(b), -1) for b in unique_bids], dtype=np.int64)
    parent_idx = np.full(m, -1, dtype=np.int64)
    has_p = parent_bid >= 0
    if has_p.any():
        parent_idx[has_p] = np.searchsorted(unique_bids, parent_bid[has_p])
    roots = np.flatnonzero(parent_idx < 0)
    root = int(roots[0]) if len(roots) else 0

    depth = np.zeros(m, dtype=np.int64)
    for _ in range(m):
        nd = np.where(parent_idx >= 0, depth[np.maximum(parent_idx, 0)] + 1, 0)
        if np.array_equal(nd, depth):
            break
        depth = nd
    max_depth = float(depth.max()) or 1.0

    # 子树骨架点数（自底向上累加），对应剪除该枝的级联规模
    sub = counts.astype(np.float64).copy()
    for lv in range(int(depth.max()), 0, -1):
        sel = np.flatnonzero(depth == lv)
        np.add.at(sub, parent_idx[sel], sub[sel])

    # 拓扑边
    t_src = np.flatnonzero(parent_idx >= 0)
    t_dst = parent_idx[t_src]
    pc_codes = np.minimum(t_src, t_dst) * m + np.maximum(t_src, t_dst)

    # 空间边：非父子分枝间骨架点最小距离 ≤ r
    r = SPATIAL_RADIUS_FACTOR * cfg.d_cross
    pairs = cKDTree(xyz).query_pairs(r, output_type="ndarray")
    s_lo = s_hi = s_d = np.zeros(0)
    if len(pairs):
        a, b = seg_id[pairs[:, 0]], seg_id[pairs[:, 1]]
        keep = a != b
        pairs, a, b = pairs[keep], a[keep], b[keep]
        dist = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        code = lo * m + hi
        o = np.lexsort((dist, code))
        first = np.r_[True, code[o][1:] != code[o][:-1]]
        sel = o[first]
        s_lo, s_hi, s_d, s_code = lo[sel], hi[sel], dist[sel], code[sel]
        keep = ~np.isin(s_code, pc_codes)
        s_lo, s_hi, s_d = s_lo[keep], s_hi[keep], s_d[keep]
        cap = MAX_SPATIAL_PER_NODE * m
        if len(s_d) > cap:
            o = np.argsort(s_d)[:cap]
            s_lo, s_hi, s_d = s_lo[o], s_hi[o], s_d[o]
    s_lo = s_lo.astype(np.int64)
    s_hi = s_hi.astype(np.int64)

    # 近邻交叉线索：d_cross 内的非父子邻枝个数与最小距离
    close = s_d <= cfg.d_cross
    n_close = (np.bincount(s_lo[close], minlength=m)
               + np.bincount(s_hi[close], minlength=m)).astype(np.float64)
    min_d = np.full(m, r)
    if len(s_d):
        np.minimum.at(min_d, s_lo, s_d)
        np.minimum.at(min_d, s_hi, s_d)

    x = np.stack([
        (center[:, 0] - origin[0]) / scale,
        (center[:, 1] - origin[1]) / scale,
        (center[:, 2] - origin[2]) / scale,
        vec[:, 0], vec[:, 1], vec[:, 2],
        length / scale,
        radius / scale,
        depth / max_depth,
        np.minimum(length / mean_len, 10.0),
        np.minimum(radius / mean_r, 10.0),
        (center[:, 0] - canopy[0]) / scale,
        (center[:, 1] - canopy[1]) / scale,
        (center[:, 2] - canopy[2]) / scale,
        np.minimum(length, 10.0),
        np.minimum(radius * 10.0, 10.0),
        np.minimum(CFG.traj.expert_volume_penalty * volume, 5.0),
        sub / n,
        np.minimum(n_close, 10.0) / 10.0,
        min_d / cfg.d_cross,
    ], axis=1).astype(np.float32)

    def _cos(i, j):
        return np.sum(vec[i] * vec[j], axis=1)

    t_cos = _cos(t_src, t_dst)
    s_cos = _cos(s_lo, s_hi)
    src = np.concatenate([t_src, t_dst, s_lo, s_hi])
    dst = np.concatenate([t_dst, t_src, s_hi, s_lo])
    n_t, n_s = len(t_src), len(s_lo)
    attr = np.zeros((2 * n_t + 2 * n_s, EDGE_FEAT_DIM), dtype=np.float32)
    attr[:2 * n_t, 0] = 1.0
    attr[:2 * n_t, 3] = np.tile(t_cos, 2)
    attr[2 * n_t:, 1] = 1.0
    attr[2 * n_t:, 2] = np.tile(s_d / cfg.d_cross, 2)
    attr[2 * n_t:, 3] = np.tile(s_cos, 2)

    g = np.array([pruning_ratio, m / 1000.0], dtype=np.float32)
    return {"x": x, "edge_index": np.stack([src, dst]).astype(np.int64),
            "edge_attr": attr, "g": g, "bids": unique_bids, "root": root}


def arrays_to_data(arr: dict) -> Data:
    data = Data(x=torch.from_numpy(np.asarray(arr["x"], dtype=np.float32)),
                edge_index=torch.from_numpy(np.asarray(arr["edge_index"], dtype=np.int64)),
                edge_attr=torch.from_numpy(np.asarray(arr["edge_attr"], dtype=np.float32)))
    data.g = torch.from_numpy(np.asarray(arr["g"], dtype=np.float32)).view(1, -1)
    data.root = torch.tensor([int(arr["root"])], dtype=torch.long)
    data.num_nodes = int(arr["x"].shape[0])
    return data


def skeleton_to_graph(sk: np.ndarray, parent_map: dict[int, int] | None = None,
                      pruning_ratio: float = 0.0) -> Data:
    """骨架 [N,6] → 分支级 PyG 图。"""
    return arrays_to_data(graph_arrays(sk, parent_map, pruning_ratio))


# ==============================================================================
# 编码器
# ==============================================================================

class BranchGraphEncoder(nn.Module):
    """边感知图注意力（TransformerConv）+ 残差 → 逐分枝向量；双池化 + 全局特征 → 状态向量。"""

    def __init__(self, in_dim: int = BRANCH_FEAT_DIM, edge_dim: int = EDGE_FEAT_DIM,
                 global_dim: int = GLOBAL_FEAT_DIM, hidden: int = 128,
                 heads: int = 4, layers: int = 3, state_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.node_dim = hidden
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                        nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(TransformerConv(hidden, hidden // heads, heads=heads,
                                              concat=True, edge_dim=edge_dim,
                                              dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden))
        self.dropout = dropout
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2 + global_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, data: Data | Batch):
        """返回 (state [G,state_dim], nodes [G,N,hidden], legal [G,N] bool)。

        nodes 第 i 列对应该图按 branch_id 升序的第 i 根分枝；legal 屏蔽
        填充位与主干（根分枝不可剪）。
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + F.elu(conv(h, edge_index, data.edge_attr)))
            h = F.dropout(h, p=self.dropout, training=self.training)
        pooled = torch.cat([global_mean_pool(h, batch),
                            global_max_pool(h, batch),
                            data.g.view(-1, data.g.size(-1))], dim=-1)
        state = self.fuse(pooled)
        nodes, legal = to_dense_batch(h, batch)
        legal = legal.clone()
        legal[torch.arange(legal.size(0), device=legal.device), data.root] = False
        return state, nodes, legal
