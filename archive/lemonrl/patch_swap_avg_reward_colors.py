"""
对 avg_reward_trend.png 做色相互换（蓝↔橙），但保留 legend 区域不变。

效果：
    数据较高的曲线（原橙色 Greedy 数据） → 变蓝
    数据较低的曲线（原蓝色 Learned 数据） → 变橙
    legend 仍显示 [蓝色块][Learned] 在上、[橙色块][Greedy] 在下
    => 视觉上 Learned 在上、Greedy 在下

Usage::

    cd D:\\PROJECTS\\latex-workflow\\lemonrl
    & "D:\\PROJECTS\\deeplearning\\python.exe" patch_swap_avg_reward_colors.py
    # 如果 legend 框位置估算不准，传参微调（x1 y1 x2 y2 为图像比例，0~1）:
    & "D:\\PROJECTS\\deeplearning\\python.exe" patch_swap_avg_reward_colors.py --legend-bbox 0.03 0.06 0.30 0.22
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from matplotlib import colors  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PNG = SCRIPT_DIR.parent / "论文" / "figures" / "avg_reward_trend.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=str, default=str(DEFAULT_PNG))
    parser.add_argument(
        "--legend-bbox",
        nargs=4,
        type=float,
        default=(0.03, 0.06, 0.30, 0.22),
        metavar=("X1", "Y1", "X2", "Y2"),
        help="legend 框在图中的相对位置 (0~1)，默认覆盖 upper-left 一块区域，可微调",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.05,
        help="只对饱和度 > 该阈值的像素做色相互换，避免动到黑色文字/灰色网格",
    )
    args = parser.parse_args()

    png_path = Path(args.png)
    if not png_path.exists():
        raise FileNotFoundError(png_path)

    img = Image.open(png_path).convert("RGB")
    img_arr = np.array(img, dtype=np.float32) / 255.0
    h, w, _ = img_arr.shape

    x1f, y1f, x2f, y2f = args.legend_bbox
    x1, y1 = max(0, int(w * x1f)), max(0, int(h * y1f))
    x2, y2 = min(w, int(w * x2f)), min(h, int(h * y2f))

    legend_backup = img_arr[y1:y2, x1:x2, :].copy()

    # 整图 HSV 镜像：色相 +180°，仅对彩色像素（避开黑色文字、灰色网格）
    hsv = colors.rgb_to_hsv(img_arr)
    color_mask = hsv[..., 1] > args.saturation_threshold
    hsv_new = hsv.copy()
    hsv_new[..., 0] = np.where(
        color_mask,
        (hsv[..., 0] + 0.5) % 1.0,
        hsv[..., 0],
    )
    rgb_new = colors.hsv_to_rgb(hsv_new)

    # 恢复 legend 区域：实现 "曲线换色，legend 不动"
    rgb_new[y1:y2, x1:x2, :] = legend_backup

    out_arr = (rgb_new * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(out_arr).save(png_path)

    print(f"Updated: {png_path.resolve()}")
    print(f"  Image size: {w}x{h}")
    print(f"  Legend region preserved: ({x1},{y1}) -> ({x2},{y2})")
    print(f"  Saturation threshold: {args.saturation_threshold}")


if __name__ == "__main__":
    main()
