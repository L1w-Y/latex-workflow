"""工作进度.md：F5 记录 30–34 收紧为 30–32。"""
import io

P = r"d:\PROJECTS\latex-workflow\工作进度.md"
NEW = """30. **接入平台**（10-12）：`reward.py` 新 KPI 与新 R_i（旧版备份 `experiments/scratch/reward_old_kpi.py.bak`）。专家动作不读 KPI，轨迹不重生成；`reward_parts.py` 分 compute（逐步复算规则项 + 六项 ΔK）/ calibrate（标定 Δe）/ apply（新回报写 `graph_cache/rtg_<set>/`，不改图缓存）三步；`load_cache`、`evaluate.py`、`train.py`、`train_offline.py` 加 `--reward-set`（默认 full）

31. **标定与权重**（10-13，`reward_mix.py`、`clip_check.py`）
   - Δe：层次 6.54e-3 / 拥挤 3.05e-3 / 平衡 1.11e-2 / 光照 5.22e-3 / 潜力 2.28e-3 / 稳定 7.22e-3（训练树 70948 刀）
   - β = 1 时 R_i 已能区分（单刀专家 +4.1、随机 +2.7，与 R_q 相关 0.1），但只占专家回报 13%；R_q 按刀累加、切碎即多得，R_i 近似状态差、切碎不多得 → 用户定 β = 3（约 32%）
   - 单步裁剪 [−3, 40] 截断专家 38.5% 的步、把保护惩罚截成 −3 → 用户定去掉
   - 结果：训练树专家回报 2204（规则分 1506 + KPI 分 702）、随机 949；DT 按枝目标系数 6.96
   - 论文需说明：随机剪法 70% 的刀 KPI 也变好（删枝普遍减拥挤、增透光），靠剪枝比上限约束

32. **新奖励下重训**（10-13）：DT 验证 top-1 0.134 / top-5 0.382（与旧奖励下持平）；CQL、BCQ、IQL 各 10000 步，Q 值收敛无发散；BC 不读回报，沿用原 checkpoint

**完成情况**：✅ 完成，转 E3 / E2
"""

with io.open(P, encoding="utf-8") as f:
    lines = f.read().split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith("30. **光照 / 拥挤度 / 结果潜力新口径**"))
end = next(i for i, l in enumerate(lines) if l.startswith("**完成情况**：六项口径已定并接入"))
lines[start:end + 1] = NEW.rstrip("\n").split("\n")
with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(lines))
print("replaced lines", start + 1, "-", end + 1)
