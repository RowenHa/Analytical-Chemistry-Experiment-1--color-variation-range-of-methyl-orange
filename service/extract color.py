from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image


def average_center_hsv(image_path: str | Path, window_size: int = 21) -> tuple[float, float, float]:
	"""
	提取图片中心附近 window_size x window_size 区域像素，求平均后返回 HSV。

	返回值范围：
	- H: 0~360
	- S: 0~100
	- V: 0~100
	"""
	if window_size <= 0:
		raise ValueError("window_size 必须为正整数")

	img = Image.open(image_path).convert("RGB")
	arr = np.array(img, dtype=np.float32) / 255.0  # 归一化到 0~1

	h, w, _ = arr.shape
	cx, cy = w // 2, h // 2
	half = window_size // 2

	x1 = max(0, cx - half)
	x2 = min(w, cx + half + 1)
	y1 = max(0, cy - half)
	y2 = min(h, cy + half + 1)

	center_patch = arr[y1:y2, x1:x2]  # shape: (ph, pw, 3)
	mean_rgb = center_patch.reshape(-1, 3).mean(axis=0)

	# colorsys 使用 RGB(0~1) -> HSV(0~1)
	h01, s01, v01 = colorsys.rgb_to_hsv(*mean_rgb.tolist())

	h_deg = h01 * 360.0
	s_pct = s01 * 100.0
	v_pct = v01 * 100.0
	return h_deg, s_pct, v_pct


def main() -> None:
	parser = argparse.ArgumentParser(description="提取图片中心附近像素平均颜色（HSV）")
	parser.add_argument("image", type=str, help="图片路径")
	parser.add_argument(
		"--window",
		type=int,
		default=21,
		help="中心区域窗口大小（默认 21，表示 21x21）",
	)
	args = parser.parse_args()

	h, s, v = average_center_hsv(args.image, args.window)
	print(f"平均HSV: H={h:.2f}°, S={s:.2f}%, V={v:.2f}%")


if __name__ == "__main__":
	main()
