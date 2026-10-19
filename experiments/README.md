# experiments 目录约定

```
experiments/
├── 审稿意见清单.md / 审稿意见修改方案.md
├── data/                 训练数据：trajectories/（轨迹 + split.pkl）、graph_cache/（图缓存 + split_val.json）
├── checkpoints_bc/       当前 BC checkpoint
├── checkpoints_dt/       当前 DT checkpoint
├── scripts/              正式脚本（产出论文数据或可复现的分析）
├── scratch/              临时脚本与一次性测试，产出也放这里；不进论文，可随时清空
├── results/
│   ├── paper/            写进论文的数据，只放定稿配置跑出的结果
│   │   ├── dataset/      数据集统计（traj_stats.json）
│   │   └── <实验编号>/<split>/   如 main/test、E2_teacher/test
│   └── dev/              调参与过程数据，不直接进论文
│       └── <实验名>/<split>/     如 pilot/val（风险试水）
└── obsolete/             作废数据与脚本，只读，不得引用
    ├── 20260930_traj_v1/      v1 轨迹生成日志（父子关系 bug）
    ├── 20261001_old_model/    旧全局编号动作头的 checkpoint 与评估结果
    └── scripts_traj_v2/       v2 轨迹生成与迁移脚本
```

## 规则

1. 评估结果由 `dt_pruning/evaluate.py` 按 `--stage` 和 `--exp` 写到 `results/<stage>/<exp>/<split>/`，每个 `eval_<tag>.csv` 旁有同名 `.meta.json`，记录命令、checkpoint、目标回报和时间。
2. 测试树只能以 `--stage paper` 运行；`paper` 阶段已有同名结果时拒绝覆盖，确需重跑加 `--overwrite`，并在 `工作进度.md` 记录原因。
3. `paper/` 实验编号与 `审稿意见修改方案.md` 对应：`main`（主对比，E3 + E6）、`E2_teacher`、`E4_ablation`、`E5_sensitivity`。
4. `dev/` 中的结论要进论文，须用定稿配置在 `paper/` 下重跑，不直接搬运。
5. `checkpoints_*` 一旦用于 `paper/` 结果即冻结；改训练配置另开目录（如 `checkpoints_dt_<说明>/`）。
6. `results/dev/pilot/val/` 为 2026-10-09 风险试水结果（DT 用 `checkpoints_dt/dt_best.pt`），早于 `.meta.json` 机制，无元数据文件；各 tag 含义见 `工作进度.md` 风险试水节。

## v2 独立流水线（2026-10-16）

v1 的 `data/trajectories/`、`data/graph_cache/` 和 `checkpoints_*` 保持只读。v2 使用：

```text
data/manifests/v2_small_manifest.json  固定20训练、10选模、31在线验证及配置血缘
data/trajectories_v2/                  30棵原始轨迹，只读保留
data/trajectories_v2_curated/          20+10精炼轨迹
data/graph_cache_v2_curated/           schema=2图、ΔK、奖励分量及RTG
checkpoints_bc_v2/                v2公平BC
checkpoints_dt_v2/                v2双头DT
results/dev/e3_redesign_v2/       v2开发验证
```

所有长任务由用户在项目根目录分阶段执行。每阶段完成后先检查终端输出和产物，再进入下一阶段。

### 0. 名单检查（短任务）

```powershell
python dt_pruning/v2_manifest.py check
```

应输出 `train=20, val=10`，manifest hash 为 `107a612683704e33`。manifest 另含31棵 `eval_val`，只用于在线评估。

### 1. 轨迹生成与精炼（已完成）

原始任务运行至30棵后停止，原文件保留在 `data/trajectories_v2/`。精炼脚本按固定种子划分20训练+10选模，并以固定操作成本20、边际净收益阈值80、连续3步低收益重建 STOP 和 RTG，输出到 `data/trajectories_v2_curated/`。不得删除或覆盖原始30棵。

### 2. 建立 schema=2 图缓存（长任务）

```powershell
python -u dt_pruning/graph_cache_v2.py --split all --workers 10
```

输出到 `data/graph_cache_v2_curated/`；支持按树续跑。完成判据：`index.json` 覆盖30棵，`meta.json` 记录节点30维、全局8维。

### 3. KPI尺度标定与回报旁路（中短任务）

```powershell
python dt_pruning/reward_parts_v2.py calibrate
python dt_pruning/reward_parts_v2.py apply --set full
```

标定只使用 manifest 的20棵训练树，10棵选模树不参与。输出 `graph_cache_v2_curated/kpi_scales.json` 与 `rtg_full/`。

### 4. 极短冒烟训练（先执行）

```powershell
python -u dt_pruning/train_bc_v2.py --epochs 1 --max-trees 2 --max-batches 2 --out checkpoints_bc_smoke_v2
python -u dt_pruning/train_v2.py --epochs 1 --max-trees 2 --max-batches 2 --eval-every 1 --out checkpoints_dt_smoke_v2
python -u dt_pruning/evaluate_v2.py --checkpoint checkpoints_dt_smoke_v2/dt_last.pt --limit 2 --exp smoke_v2 --tag dt_smoke
```

冒烟只检查 shape、损失、双头推理和落盘，不解释数值。

### 5. 正式训练（长任务；冒烟核验后再运行）

```powershell
python -u dt_pruning/train_bc_v2.py --epochs 30 --batch-size 64 --out checkpoints_bc_v2
python -u dt_pruning/train_v2.py --epochs 60 --context-len 16 --batch-size 8 --batches-per-epoch 200 --out checkpoints_dt_v2
```

### 6. 31棵验证树自主停止评估（长任务）

```powershell
python -u dt_pruning/evaluate_v2.py --checkpoint checkpoints_dt_v2/dt_best.pt --exp e3_redesign_v2 --tag dt_v2
```

结果写入 `results/dev/e3_redesign_v2/val/`。该阶段不使用 STOP bias，也不固定最低剪枝量。首先检查自主剪枝比、模型自主 STOP 比例、终点六维 KPI 和安全动作，再决定是否进入公平 CQL/BCQ/IQL 与 E2–E4。
