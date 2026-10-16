"""
直接用 PIL 修改 avg_reward_trend.png 的标题文字，不需要重跑 benchmark。

Usage::

    cd D:\\PROJECTS\\latex-workflow\\lemonrl
    & "D:\\PROJECTS\\deeplearning\\python.exe" patch_avg_reward_title.py
    & "D:\\PROJECTS\\deeplearning\\python.exe" patch_avg_reward_title.py "其它标题"

可选参数:
    title                 新标题文本（默认使用 "Average step reward over the full test set..."）
    --png  PATH           目标 PNG（默认 ../论文/figures/avg_reward_trend.png）
    --font-size N         字号像素（默认 22）
    --title-region FRAC   标题区域占图高的比例（默认 0.07）
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from matplotlib import font_manager  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PNG = SCRIPT_DIR.parent / "论文" / "figures" / "avg_reward_trend.png"
DEFAULT_TITLE = "Average step reward over the full test set (mean ± 1 std)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("title", nargs="?", default=DEFAULT_TITLE)
    parser.add_argument("--png", type=str, default=str(DEFAULT_PNG))
    parser.add_argument("--font-size", type=int, default=22)
    parser.add_argument("--title-region", type=float, default=0.07,
                        help="标题区域占图高度的比例，越大遮挡范围越大")
    args = parser.parse_args()

    png_path = Path(args.png)
    if not png_path.exists():
        raise FileNotFoundError(f"PNG not found: {png_path}")

    img = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    title_h = int(h * args.title_region)
    draw.rectangle([(0, 0), (w, title_h)], fill="white")

    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    font = ImageFont.truetype(font_path, size=args.font_size)

    bbox = draw.textbbox((0, 0), args.title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (w - tw) // 2
    y = max(0, (title_h - th) // 2 - 4)
    draw.text((x, y), args.title, font=font, fill="black")

    img.save(png_path)
    print(f"Updated title in: {png_path.resolve()}")
    print(f"  New title: {args.title!r}")
    print(f"  Image size: {w}x{h}  | title region height: {title_h}px  | font size: {args.font_size}px")


if __name__ == "__main__":
    main()
