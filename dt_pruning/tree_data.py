"""树数据底座：骨架读取、分枝图抽象、子树删除、三级安全约束。

骨架文件格式（skeleton.txt）：每行 6 列
    x  y  z  radius  level(层级深度, 1基: 主干=1, 一级侧枝=2, ...)  branch_id
    注：第 5 列字段原名 parent_point_id，经数据实测语义为「分枝层级深度」
    （每个分枝内恒定、取值 [1,5]），并非父点行号；骨架不含显式父分枝字段。
父子关系重建规则（全平台唯一实现，见 build_parent_map）：
    分枝 b 的父分枝 = 层级严格更低的分枝中、与 b 的基部点几何最近的那个；
    层级=1 的分枝（唯一，即主干）无父，视为根。
    层级严格递减保证无环；任何 level≥2 分枝都能挂到父，杜绝"幻影根"。
    基部点 = 分枝内离树根（min-z 点）最近的骨架点。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


# 列索引
X, Y, Z, R, PARENT, BRANCH = range(6)


# ==============================================================================
# 读取
# ==============================================================================

def load_skeleton(path: str) -> np.ndarray:
    """读取 skeleton.txt，返回 [N, 6] 浮点数组。"""
    sk = np.loadtxt(path, dtype=np.float64)
    if sk.ndim == 1:
        sk = sk[None, :]
    if sk.shape[1] != 6:
        raise ValueError(f"骨架文件列数应为 6，实际 {sk.shape[1]}: {path}")
    return sk


def collect_tree_dirs(data_root: str) -> list[str]:
    """收集数据根目录下所有 LemonXX_NN 树文件夹，按名称排序。"""
    pattern = re.compile(r"^Lemon\d{2}_\d+$")
    dirs = [d for d in os.listdir(data_root)
            if pattern.match(d) and os.path.isdir(os.path.join(data_root, d))]
    dirs.sort()
    return [os.path.join(data_root, d) for d in dirs]


def tree_name(tree_dir: str) -> str:
    return os.path.basename(tree_dir.rstrip("/\\"))


def skeleton_path(tree_dir: str) -> str:
    return os.path.join(tree_dir, "skeleton", "skeleton.txt")


# ==============================================================================
# 分枝图
# ==============================================================================

def branch_ids(sk: np.ndarray) -> np.ndarray:
    """当前骨架中的全部分枝 ID（去重排序）。"""
    return np.unique(sk[:, BRANCH].astype(np.int64))


def trunk_branch_id(sk: np.ndarray) -> int:
    """主干分枝 ID：唯一的 level=1 分枝；缺失时退化为 min-z 点所在分枝。"""
    lv = sk[:, PARENT].astype(np.int64)
    br = sk[:, BRANCH].astype(np.int64)
    trunks = np.unique(br[lv == 1])
    if len(trunks) == 1:
        return int(trunks[0])
    return int(br[np.argmin(sk[:, Z])])


def build_parent_map(sk: np.ndarray) -> dict[int, int]:
    """{分枝ID: 父分枝ID}；根分枝（主干）的父为 -1。

    全平台唯一父子关系实现：
    父 = 层级严格更低的分枝中、与基部点几何最近的那个（按层级分组建
    KDTree，逐组批量查询）；层级严格递减 → 天然无环、无幻影根。
    """
    bids = [int(b) for b in branch_ids(sk)]
    br = sk[:, BRANCH].astype(np.int64)
    lv = sk[:, PARENT].astype(np.int64)

    level: dict[int, int] = {}
    base_pt: dict[int, np.ndarray] = {}
    root_xyz = sk[np.argmin(sk[:, Z]), :3]
    for b in bids:
        rows = np.nonzero(br == b)[0]
        level[b] = int(lv[rows[0]])
        pts = sk[rows, :3]
        base_pt[b] = pts[np.argmin(np.sum((pts - root_xyz) ** 2, axis=1))]

    # 每个层级一棵 KDTree（点位级）
    level_trees: dict[int, tuple[cKDTree, np.ndarray]] = {}
    for L in np.unique(lv):
        rows = np.nonzero(lv == L)[0]
        level_trees[int(L)] = (cKDTree(sk[rows, :3]), br[rows])

    parent_map: dict[int, int] = {}
    for b in bids:
        L = level[b]
        if L <= 1:
            parent_map[b] = -1
            continue
        best_d, best_parent = np.inf, -1
        for lower, (tree, owners) in level_trees.items():
            if lower >= L:
                continue
            d, idx = tree.query(base_pt[b])
            if d < best_d:
                best_d, best_parent = float(d), int(owners[int(idx)])
        # 兜底：无更低层级（数据异常）时挂到主干，不成为幻影根
        parent_map[b] = best_parent if best_parent >= 0 else trunk_branch_id(sk)
    return parent_map


def build_children_map(parent_map: dict[int, int]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for bid, pbid in parent_map.items():
        if pbid >= 0:
            children.setdefault(pbid, []).append(bid)
    return children


def descendants(children: dict[int, list[int]], bid: int) -> set[int]:
    """分枝 bid 的全部后代（不含自身），BFS。"""
    out: set[int] = set()
    stack = list(children.get(bid, []))
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack.extend(children.get(cur, []))
    return out


def root_branches(parent_map: dict[int, int]) -> set[int]:
    return {bid for bid, pbid in parent_map.items() if pbid < 0}


# ==============================================================================
# 树状态
# ==============================================================================

@dataclass
class TreeState:
    """剪枝过程中的树状态。

    skeleton 随剪枝更新；拓扑映射在每次删除后重建（N 千级，开销可忽略）。
    """
    skeleton: np.ndarray
    initial_nodes: int                       # 初始骨架点数 |S_0|
    initial_branch_count: int                # 初始分枝数 N_0
    parent_map: dict[int, int] = field(init=False)
    children: dict[int, list[int]] = field(init=False)

    def __post_init__(self):
        self.refresh_topology()

    @classmethod
    def from_skeleton(cls, sk: np.ndarray) -> "TreeState":
        return cls(skeleton=sk.copy(),
                   initial_nodes=sk.shape[0],
                   initial_branch_count=len(branch_ids(sk)))

    def refresh_topology(self):
        self.parent_map = build_parent_map(self.skeleton)
        self.children = build_children_map(self.parent_map)

    # ---- 安全约束（论文 sec:model:preprocess：三级）----
    def can_remove(self, bid: int, min_nodes_abs: int = 10,
                   min_nodes_ratio: float = 0.1) -> bool:
        """三级安全约束：根分枝保护 + 最小结构约束 + 完整性检查。"""
        if bid not in self.parent_map:
            return False
        # (i) 根分枝保护
        if self.parent_map[bid] < 0:
            return False
        # (ii) 最小结构约束：|S_{t+1}| >= max(10, 0.1 * |S_0|)
        remove_set = {bid} | descendants(self.children, bid)
        remaining = self.skeleton[~np.isin(self.skeleton[:, BRANCH], list(remove_set))]
        threshold = max(min_nodes_abs, int(np.floor(min_nodes_ratio * self.initial_nodes)))
        if remaining.shape[0] < threshold:
            return False
        # (iii) 完整性检查：所有根分枝在剪后必须仍然存在
        roots = root_branches(self.parent_map)
        if roots & remove_set:
            return False
        if not roots.issubset(set(np.unique(remaining[:, BRANCH].astype(np.int64)))):
            return False
        return True

    def legal_actions(self, min_nodes_abs: int = 10,
                      min_nodes_ratio: float = 0.1) -> list[int]:
        """当前可剪分枝集合。"""
        return [int(bid) for bid in branch_ids(self.skeleton)
                if self.can_remove(int(bid), min_nodes_abs, min_nodes_ratio)]

    def remove_branch(self, bid: int):
        """执行子树剪除：移除 bid 及其全部后代，并重建拓扑。"""
        remove_set = {bid} | descendants(self.children, bid)
        mask = ~np.isin(self.skeleton[:, BRANCH], list(remove_set))
        self.skeleton = self.skeleton[mask]
        self.refresh_topology()

    def pruning_ratio(self) -> float:
        """累计剪枝比 η = (N_0 - N_f) / N_0。"""
        return (self.initial_branch_count - len(branch_ids(self.skeleton))) \
            / max(self.initial_branch_count, 1)
