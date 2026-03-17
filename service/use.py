from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def hsv_to_features(h: float, s: float, v: float) -> np.ndarray:
	"""将 HSV(H:0~360, S/V:0~100) 转为训练时使用的特征。"""
	h_rad = np.deg2rad(h)
	return np.array([[np.sin(h_rad), np.cos(h_rad), s, v]], dtype=np.float64)


def main() -> None:
	parser = argparse.ArgumentParser(description="使用已训练 SVR 模型预测 pH")
	parser.add_argument("--model", type=str, default="model1.pkl", help="模型文件路径")
	parser.add_argument("--h", type=float, required=True, help="色相 H（0~360）")
	parser.add_argument("--s", type=float, required=True, help="饱和度 S（0~100）")
	parser.add_argument("--v", type=float, required=True, help="明度 V（0~100）")
	args = parser.parse_args()

	model_path = Path(args.model)
	if not model_path.exists():
		raise FileNotFoundError(f"模型文件不存在: {model_path}")

	with model_path.open("rb") as f:
		model = pickle.load(f)

	x = hsv_to_features(args.h, args.s, args.v)
	y_pred = float(model.predict(x)[0])

	print(f"输入HSV: H={args.h:.2f}°, S={args.s:.2f}%, V={args.v:.2f}%")
	print(f"预测 pH: {y_pred:.4f}")


if __name__ == "__main__":
	main()
