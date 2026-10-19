"""v2 六维 KPI 标定尺度的唯一运行时入口。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from config import CFG
from reward import KPI_KEYS


SCALES_PATH = (
    Path(CFG.data.work_root) / CFG.data.v2_cache_dir / "kpi_scales.json").resolve()


@lru_cache(maxsize=1)
def load_v2_kpi_scales() -> dict[str, float]:
    if not SCALES_PATH.exists():
        raise FileNotFoundError(f"缺少v2 KPI尺度：{SCALES_PATH}")
    with SCALES_PATH.open(encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("schema_version") != 2:
        raise ValueError(f"KPI尺度schema错误：{meta.get('schema_version')}")
    scales = meta.get("scales", {})
    if set(scales) != set(KPI_KEYS):
        raise ValueError(f"KPI尺度维度错误：{sorted(scales)}")
    values = {k: float(scales[k]) for k in KPI_KEYS}
    if not all(np.isfinite(x) and x > 0 for x in values.values()):
        raise ValueError(f"KPI尺度必须为有限正数：{values}")
    return values


def activate_v2_kpi_scales() -> dict[str, float]:
    """将标定尺度注入统一奖励配置，并返回相同字典。"""
    scales = load_v2_kpi_scales()
    CFG.reward.kpi_scales = dict(scales)
    return scales


def scale_vector() -> np.ndarray:
    scales = load_v2_kpi_scales()
    return np.asarray([scales[k] for k in KPI_KEYS], dtype=np.float32)
