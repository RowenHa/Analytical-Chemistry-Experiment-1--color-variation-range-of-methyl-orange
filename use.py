from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def hsv_to_features(h: float, s: float, v: float) -> np.ndarray:
	"""将 HSV(H:0~360, S/V:0~100) 转为训练时使用的特征。"""
	h_rad = np.deg2rad(h)
	return np.array([[np.sin(h_rad), np.cos(h_rad), s, v]], dtype=np.float64)


def rgb_to_features(r: float, g: float, b: float) -> np.ndarray:
	"""将 RGB(0~255) 转为训练时使用的特征。"""
	return np.array([[r, g, b]], dtype=np.float64)


def main() -> None:
	parser = argparse.ArgumentParser(description="使用已训练 SVR 模型预测 pH")
	parser.add_argument("--model", type=str, default="model1.pkl", help="模型文件路径")
	parser.add_argument("--feature", type=str, choices=["hsv", "rgb"], default="hsv", help="预测特征类型")
	parser.add_argument("--h", type=float, default=None, help="色相 H（0~360）")
	parser.add_argument("--s", type=float, default=None, help="饱和度 S（0~100）")
	parser.add_argument("--v", type=float, default=None, help="明度 V（0~100）")
	parser.add_argument("--r", type=float, default=None, help="红色通道 R（0~255）")
	parser.add_argument("--g", type=float, default=None, help="绿色通道 G（0~255）")
	parser.add_argument("--b", type=float, default=None, help="蓝色通道 B（0~255）")
	parser.add_argument("--json", dest="as_json", action="store_true", help="以 JSON 输出预测结果")
	args = parser.parse_args()

	model_path = Path(args.model)
	if not model_path.exists():
		raise FileNotFoundError(f"模型文件不存在: {model_path}")

	with model_path.open("rb") as f:
		model = pickle.load(f)

	if args.feature == "hsv":
		if args.h is None or args.s is None or args.v is None:
			raise ValueError("feature=hsv 时必须提供 --h --s --v")
		x = hsv_to_features(args.h, args.s, args.v)
	else:
		if args.r is None or args.g is None or args.b is None:
			raise ValueError("feature=rgb 时必须提供 --r --g --b")
		x = rgb_to_features(args.r, args.g, args.b)

	y_pred = float(model.predict(x)[0])

	if args.as_json:
		payload = {
			"model": model_path.name,
			"feature": args.feature,
			"ph": y_pred,
		}
		if args.feature == "hsv":
			payload["h"] = float(args.h)
			payload["s"] = float(args.s)
			payload["v"] = float(args.v)
		else:
			payload["r"] = float(args.r)
			payload["g"] = float(args.g)
			payload["b"] = float(args.b)

		print(
			json.dumps(
				payload,
				ensure_ascii=False,
			)
		)
		return

	if args.feature == "hsv":
		print(f"输入HSV: H={args.h:.2f}°, S={args.s:.2f}%, V={args.v:.2f}%")
	else:
		print(f"输入RGB: R={args.r:.2f}, G={args.g:.2f}, B={args.b:.2f}")
	print(f"预测 pH: {y_pred:.4f}")


if __name__ == "__main__":
	main()
