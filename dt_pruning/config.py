"""dt_pruning 全局配置：所有超参数集中于此，全平台唯一来源。

参数口径以论文《论文0606.tex》为准；与旧代码不一致处（如奖励裁剪 [-3,40]）以论文为准。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ==============================================================================
# 奖励与 KPI（论文第 3 节）
# ==============================================================================

@dataclass
class RewardConfig:
    # --- 复合奖励权重（论文 eq:reward_total）；各分项自身有界，不再整体裁剪 ---
    alpha: float = 5.0                 # R_q 权重
    beta: float = 3.0                  # R_i 权重
    gamma: float = 1.0                 # R_c 权重

    # --- 交叉枝判别（论文 tab:param_structure）---
    d_cross: float = 0.15              # 空间邻近阈值 (m)
    theta_min: float = 30.0            # 方向角下限 (度)
    theta_max: float = 150.0           # 方向角上限 (度)
    o_min: float = 0.20                # 水平投影重叠下限
    sigma_min: float = 0.30            # 严重度阈值

    # --- 内膛枝 ---
    tau_in: float = -0.2
    l_in: float = 0.15                 # (m)

    # --- 徒长枝 ---
    alpha_vig: float = 30.0            # 与竖直方向夹角上限 (度)
    lambda_vig: float = 1.5            # 相对树均长的倍数

    # --- 下垂枝 ---
    alpha_drp: float = 110.0           # (度)

    # --- 细弱枝 ---
    l_wk: float = 0.15                 # 长度阈值 (m)
    kappa_wk: float = 0.5              # 半径相对树均值的比例

    # --- 功能枝保护 ---
    kappa_st: float = 1.2              # 粗壮枝半径比例
    lambda_ms_lo: float = 0.3          # 中短枝相对长度下限（论文 eq:ms_protect，相对树均长）
    lambda_ms_hi: float = 0.8          # 中短枝相对长度上限

    # --- 结果潜力（论文 eq:angle_factor / eq:length_factor；δ 由代码系数 1.2/0.5/0.4 折算）---
    theta_fav: tuple = (0.0, 45.0)     # 有利仰角区间 (度)
    theta_vig: float = 60.0            # 过直立界 (度)
    theta_drp: float = -30.0           # 明显下垂界 (度)
    delta_fav: float = 0.2
    delta_vig: float = 0.5
    delta_drp: float = 0.6
    len_lambda1: float = 0.30          # 有利长度下限 (m)
    len_lambda2: float = 0.80          # 有利长度上限 (m)
    len_lambda3: float = 1.0           # 过长枝界 (m)
    delta_mid: float = 0.2
    delta_long: float = 0.3

    # --- 枝条质量奖励权重（论文 tab:quality_reward）---
    omega_cross_inferior: float = 3.0  # 交叉对中较差枝（×严重度）
    omega_cross_other: float = 1.5     # 交叉对中另一枝（×严重度）
    omega_in: float = 2.5
    omega_vig: float = 2.5
    omega_drp: float = 1.5
    omega_wk: float = 0.8
    pi_ms: float = 2.0                 # 误剪中短枝惩罚
    pi_st: float = 2.5                 # 误剪粗壮枝惩罚

    # --- KPI 权重与归一化尺度 Δe（R_i = Σ W_k·ΔK_k/Δe_k，不逐刀截断，整条轨迹求和即剪前→剪后总变化）---
    # 权重沿用论文 tab:param_reward；Δe 按训练树轨迹单刀 |ΔK| 的 95 分位标定（reward_parts.py calibrate）
    kpi_weights: dict = field(default_factory=lambda: {
        "hierarchy": 5.0, "crowding": 10.0, "balance": 8.0,
        "light": 8.0, "potential": 2.0, "stability": 5.0})
    kpi_scales: dict = field(default_factory=lambda: {
        "hierarchy": 6.54e-03, "crowding": 3.05e-03, "balance": 1.11e-02,
        "light": 5.22e-03, "potential": 2.28e-03, "stability": 7.22e-03})

    # --- R_q 按剪除量缩放：R_q × 本刀剪除点数 / rq_volume_ref（同样修剪量切成几刀得分相同）---
    rq_by_volume: bool = True
    rq_volume_ref: float = 23.5        # 训练树专家轨迹每刀平均剪除点数（reward_parts.py calibrate）

    # --- KPI 口径参数（2026-10-12 修订）---
    light_r_leaf: float = 0.10         # 叶幕包络半径 (m)，遮挡半径 = 枝半径 + 该值
    light_n_targets: int = 400         # 受光点数（初始树有效枝骨架点中固定采样）
    light_zenith: float = 30.0         # 斜向光线天顶角 (度)，另加竖直方向
    crowd_radius: float = 0.20         # 局部拥挤判定半径 d_c (m)
    w_layer: float = 0.7               # 层次性中分层负载占比的权重（其余为骨干枝排列 H_main）

    # --- 约束惩罚 ---
    pi_main: float = 5.0               # 误剪骨干枝（L>0.5 且 r>0.02）
    main_branch_length: float = 0.5
    main_branch_radius: float = 0.02
    eta_v: float = 0.15                # 单步最大剪除体积比
    lambda_v: float = 10.0

    # --- 终止奖励（论文 eq:stop_reward）---
    eta_lo: float = 0.18
    eta_hi: float = 0.32
    eta_buf_lo: float = 0.10
    eta_buf_hi: float = 0.40
    eta_under: float = 0.08
    eta_over: float = 0.45
    r_ideal: float = 3.0
    r_buf: float = 1.0
    r_under: float = 0.5
    r_over: float = 1.5

    # --- 平衡系数（偏移按冠幅归一）---
    k_b: float = 2.0

    # --- 有效枝长度下限 ---
    l_eff: float = 0.05                # 有效枝 (m)
    eps: float = 1e-6


# ==============================================================================
# 数据与轨迹（论文第 4 节）
# ==============================================================================

@dataclass
class DataConfig:
    data_root: str = r"D:\DownLoad\baiduyunDownload\9lemon\00lemon"
    # 实验数据（轨迹/checkpoint/评估结果）统一输出到 experiments/，平台代码留在 dt_pruning/
    work_root: str = r"d:\PROJECTS\latex-workflow\experiments"
    train_ratio: float = 0.85          # 树级 85/15 划分
    split_seed: int = 42

    # 三级安全约束（论文 sec:model:preprocess）
    min_nodes_abs: int = 10
    min_nodes_ratio: float = 0.1       # 相对初始骨架点数


@dataclass
class TrajectoryConfig:
    n_expert: int = 1                  # 每树专家轨迹数
    n_random: int = 4                  # 每树随机轨迹数
    random_target_lo: float = 0.20     # 随机轨迹目标剪枝比区间
    random_target_hi: float = 0.30
    max_pruning_ratio: float = 0.30    # 硬上限
    # 专家目标剪枝比按树轮转分档（解决"专家全部顶到上限、STOP 监督单调"，
    # 并建立"目标回报—剪枝强度"梯度；v2 口径，见工作进度记录 9/12）
    expert_target_tiers: tuple = (0.16, 0.20, 0.23, 0.26, 0.30)
    expert_volume_penalty: float = 50.0  # 专家策略体积惩罚系数 λ_exp
    seed: int = 42
    stop_action: int = -1              # 轨迹文件中 STOP 的标记


# ==============================================================================
# 模型与训练（论文 tab:dt_config；状态维度按决议改为 128）
# ==============================================================================

@dataclass
class ModelConfig:
    state_dim: int = 128               # 图编码器输出维度
    hidden_dim: int = 128              # DT 隐维
    n_layer: int = 3
    n_head: int = 4
    dropout: float = 0.1
    # 图编码器
    gnn_hidden: int = 128
    gnn_heads: int = 4
    gnn_layers: int = 3


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.10
    min_lr_ratio: float = 0.1          # 余弦衰减下限：峰值学习率的比例（不衰减到 0）
    grad_clip: float = 1.0
    epochs: int = 1000
    max_len_quantile: float = 0.95     # 序列长度取轨迹长度 95 分位数
    log_every: int = 20
    save_every: int = 100
    seed: int = 42
    device: str = "cuda"               # 无 GPU 时自动回退 cpu


@dataclass
class EvalConfig:
    max_steps: int = 400               # rollout 步数上限
    target_return_quantile: float = 1.0  # 初始目标回报取训练树专家统计量的该分位数（1.0 即最大值）
    greedy_threshold: float = 0.45     # 贪心基线停止阈值 τ_g
    greedy_alpha0: float = 35.0        # 贪心基线竖直角参照 α_0 (度)
    greedy_weights: tuple = (1.0, 1.0, 1.0)  # w1, w2, w3
    seed: int = 42


# ==============================================================================

@dataclass
class Config:
    reward: RewardConfig = field(default_factory=RewardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    traj: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


CFG = Config()
