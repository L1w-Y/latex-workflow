"""v2 模块消融的统一特征开关。"""
from __future__ import annotations

from torch_geometric.data import Batch, Data


def mask_graph_features(graph: Data | Batch, *, use_rule_features: bool,
                        use_kpi_features: bool) -> Data | Batch:
    """原位屏蔽被消融模块，保持网络维度和其余结构完全一致。"""
    if not use_rule_features:
        # schema=2 节点前20维为基础几何/拓扑，后10维为园艺规则特征。
        graph.x[:, 20:30] = 0
    if not use_kpi_features:
        # 全局前2维为剪枝比/分枝数，后6维为当前六维 KPI。
        graph.g[:, 2:8] = 0
    return graph
