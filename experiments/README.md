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
