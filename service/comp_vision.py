from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_hsv_array(rgb01: np.ndarray) -> np.ndarray:
	"""将 RGB(0~1) 数组转换为 HSV(0~1) 数组。"""
	r = rgb01[..., 0]
	g = rgb01[..., 1]
	b = rgb01[..., 2]

	cmax = np.max(rgb01, axis=-1)
	cmin = np.min(rgb01, axis=-1)
	delta = cmax - cmin

	h = np.zeros_like(cmax)
	s = np.zeros_like(cmax)
	v = cmax

	nonzero = delta > 1e-12
	s[nonzero] = delta[nonzero] / cmax[nonzero]

	idx = (cmax == r) & nonzero
	h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0

	idx = (cmax == g) & nonzero
	h[idx] = (b[idx] - r[idx]) / delta[idx] + 2.0

	idx = (cmax == b) & nonzero
	h[idx] = (r[idx] - g[idx]) / delta[idx] + 4.0

	h = h / 6.0
	return np.stack([h, s, v], axis=-1)


def hsv_distance(p1: np.ndarray, p2: np.ndarray) -> float:
	"""HSV 距离：考虑 H 的环形特性。输入范围均为 0~1。"""
	dh = abs(float(p1[0]) - float(p2[0]))
	dh = min(dh, 1.0 - dh)  # hue 环形距离
	ds = float(p1[1]) - float(p2[1])
	dv = float(p1[2]) - float(p2[2])

	# 给 hue 略高权重，增强“颜色相近”约束
	return float(np.sqrt((2.0 * dh) ** 2 + ds**2 + dv**2))


def collect_similar_pixels(
	image_path: str | Path,
	target_count: int = 500,
	threshold: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], int]:
	"""
	从图片中心开始，按邻域扩张，收集和中心像素颜色相近的像素，最多 target_count 个。

	返回:
	- mean_rgb_255: 平均 RGB (0~255)
	- mean_hsv_hsv: 平均 HSV (H:0~360, S:0~100, V:0~100)
	- center_xy: 中心像素坐标 (x, y)
	- picked_count: 实际收集像素数
	"""
	if target_count <= 0:
		raise ValueError("target_count 必须为正整数")

	img = Image.open(image_path).convert("RGB")
	rgb = np.asarray(img, dtype=np.float32) / 255.0
	hsv = rgb_to_hsv_array(rgb)

	h, w, _ = rgb.shape
	cx, cy = w // 2, h // 2

	center_hsv = hsv[cy, cx]

	visited = np.zeros((h, w), dtype=bool)
	q: deque[tuple[int, int]] = deque()
	q.append((cx, cy))
	visited[cy, cx] = True

	picked: list[tuple[int, int]] = []

	while q and len(picked) < target_count:
		x, y = q.popleft()
		d = hsv_distance(hsv[y, x], center_hsv)
		if d <= threshold:
			picked.append((x, y))

			# 只有当前点“相近”时，继续向外扩张，保证是中心附近连通区域
			for nx, ny in (
				(x - 1, y),
				(x + 1, y),
				(x, y - 1),
				(x, y + 1),
				(x - 1, y - 1),
				(x + 1, y - 1),
				(x - 1, y + 1),
				(x + 1, y + 1),
			):
				if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
					visited[ny, nx] = True
					q.append((nx, ny))

	if not picked:
		raise RuntimeError("在当前阈值下未找到相近像素，请适当增大 threshold")

	ys = np.array([p[1] for p in picked], dtype=np.int32)
	xs = np.array([p[0] for p in picked], dtype=np.int32)

	picked_rgb = rgb[ys, xs]  # 0~1
	mean_rgb01 = picked_rgb.mean(axis=0)
	mean_rgb_255 = mean_rgb01 * 255.0

	# 平均 RGB 再转 HSV（避免直接对 hue 做算术平均导致跨 360° 失真）
	mean_hsv01 = rgb_to_hsv_array(mean_rgb01.reshape(1, 1, 3))[0, 0]
	mean_hsv = np.array([mean_hsv01[0] * 360.0, mean_hsv01[1] * 100.0, mean_hsv01[2] * 100.0])

	return mean_rgb_255, mean_hsv, (cx, cy), len(picked)


def main() -> None:
	parser = argparse.ArgumentParser(description="从图像中心寻找相近像素（最多500个）并求平均颜色")
	parser.add_argument("image", type=str, help="图片路径")
	parser.add_argument("--count", type=int, default=500, help="目标像素个数，默认 500")
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.10,
		help="HSV 相似阈值（默认 0.10，建议范围 0.06~0.20）",
	)
	args = parser.parse_args()

	mean_rgb, mean_hsv, center_xy, picked_count = collect_similar_pixels(
		image_path=args.image,
		target_count=args.count,
		threshold=args.threshold,
	)

	print(f"中心像素坐标: {center_xy}")
	print(f"已收集像素数: {picked_count}")
	print(f"平均RGB: R={mean_rgb[0]:.2f}, G={mean_rgb[1]:.2f}, B={mean_rgb[2]:.2f}")
	print(f"平均HSV: H={mean_hsv[0]:.2f}°, S={mean_hsv[1]:.2f}%, V={mean_hsv[2]:.2f}%")


if __name__ == "__main__":
	main()
