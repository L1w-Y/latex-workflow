# LemonRL

LemonRL is a graph-based pruning workflow for lemon tree data, including:

- trajectory generation from raw tree skeleton/point cloud data
- GNN-based ranking/policy training in notebooks
- KPI/reward-based pruning evaluation

## Project Structure

- `generate_trajectories.py`: generate expert/random pruning trajectories
- `utils.py`: KPI and reward computation
- `gnn_rl_core.py`: PyG graph conversion and model building blocks
- `graph_lemontree.ipynb`: main training/benchmark notebook
- `data/`: raw trees and generated trajectories
- `checkpoints/`: saved model checkpoints

## Environment

A `requirements.txt` is provided from the `lemonrl` environment.

Create and install a fresh environment:

```bash
conda create -n lemonrl python=3.12 
conda activate lemonrl
pip install -r requirements.txt
```

## Generate Trajectories

```bash
python generate_trajectories.py \
  --data-root ./data \
  --output-root ./data/trajectories \
  --random-per-tree 4 \
  --max-pruning-ratio 0.30
```

Useful expert-policy controls:
- `--quick-filter-keep-ratio` (default `0.40`)
- `--quick-filter-min-candidates` (default `8`)

## Train / Benchmark

Open and run:

- `graph_lemontree.ipynb`

Recommended order:

1. dataset loading and split
2. candidate ranker training
3. benchmark (learned vs greedy)

## Notes

- Trajectory and benchmark behavior depends on data quality in `data/LemonXX_YY`.
- Large trees can still take time even after KPI/reward optimizations.

---

# LemonRL

LemonRL 是一个面向柠檬树数据的基于图的剪枝工作流，包括：

- 从原始树骨架 / 点云数据生成轨迹
- 在 notebook 中进行基于 GNN 的排序 / 策略训练
- 基于 KPI / reward 的剪枝评估

## 项目结构

- `generate_trajectories.py`：生成 expert/random 剪枝轨迹
- `utils.py`：KPI 和 reward 计算
- `gnn_rl_core.py`：PyG 图转换和模型构建模块
- `graph_lemontree.ipynb`：主要训练 / benchmark notebook
- `data/`：原始树数据和生成的轨迹
- `checkpoints/`：保存的模型 checkpoint

## 环境

已提供来自 `lemonrl` 环境的 `requirements.txt`。

创建并安装一个新的环境：

```bash
conda create -n lemonrl python=3.12 
conda activate lemonrl
pip install -r requirements.txt
```

## 生成轨迹

```bash
python generate_trajectories.py \
  --data-root ./data \
  --output-root ./data/trajectories \
  --random-per-tree 4 \
  --max-pruning-ratio 0.30
```

有用的 expert-policy 控制项：
- `--quick-filter-keep-ratio`（默认 `0.40`）
- `--quick-filter-min-candidates`（默认 `8`）

## 训练 / Benchmark

打开并运行：

- `graph_lemontree.ipynb`

推荐顺序：

1. 数据集加载和划分
2. 候选排序器训练
3. benchmark（learned vs greedy）

## 说明

- 轨迹和 benchmark 的表现依赖于 `data/LemonXX_YY` 中的数据质量。
- 即使做了 KPI / reward 优化，大树仍然可能需要较长时间处理。
