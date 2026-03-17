from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Literal, cast

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


EXPECTED_HEADER = "h,s,v,r,g,b,ph"


def parse_data(
	file_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""严格解析 H,S,V,R,G,B,pH 七列数据。"""
	path = Path(file_path)
	raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
	if not raw_lines:
		raise ValueError("数据文件为空")

	header = raw_lines[0].lower().replace(" ", "")
	if header != EXPECTED_HEADER:
		raise ValueError("数据文件头必须为 H,S,V,R,G,B,pH")

	rows: list[tuple[float, float, float, float, float, float, float]] = []
	for line_idx, row in enumerate(csv.reader(raw_lines[1:]), start=2):
		if len(row) != 7:
			raise ValueError(f"第 {line_idx} 行列数错误，期望 7 列")
		try:
			h = float(row[0])
			s = float(row[1])
			v = float(row[2])
			r = float(row[3])
			g = float(row[4])
			b = float(row[5])
			ph = float(row[6])
		except ValueError as exc:
			raise ValueError(f"第 {line_idx} 行包含非数字字段") from exc
		rows.append((h, s, v, r, g, b, ph))

	if not rows:
		raise ValueError("未在数据文件中解析到有效样本")

	arr = np.array(rows, dtype=np.float64)
	return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]


def build_features(
	feature: Literal["hsv", "rgb"],
	h_deg: np.ndarray,
	s: np.ndarray,
	v: np.ndarray,
	r: np.ndarray,
	g: np.ndarray,
	b: np.ndarray,
) -> np.ndarray:
	if feature == "hsv":
		h_rad = np.deg2rad(h_deg)
		# 色相是环形变量，用 sin/cos 编码避免 0° 与 360° 的断点问题
		return np.column_stack([np.sin(h_rad), np.cos(h_rad), s, v])
	return np.column_stack([r, g, b])


def build_model(
	c: float = 10.0,
	epsilon: float = 0.05,
	gamma: float | Literal["scale", "auto"] = "scale",
) -> Pipeline:
	model = Pipeline(
		[
			("scaler", StandardScaler()),
			("svr", SVR(kernel="rbf", C=c, epsilon=epsilon, gamma=gamma)),
		]
	)
	return model


def parse_gamma(value: str) -> float | Literal["scale", "auto"]:
	value = value.strip().lower()
	if value in {"scale", "auto"}:
		return cast(Literal["scale", "auto"], value)
	return float(value)


def auto_tune_model(x: np.ndarray, y: np.ndarray) -> tuple[Pipeline, dict[str, object], float]:
	"""使用留一法 + 网格搜索自动调参，返回最佳模型与参数。"""
	base_model = build_model()
	param_grid: dict[str, list[object]] = {
		"svr__C": [0.1, 1.0, 10.0, 50.0, 100.0],
		"svr__epsilon": [0.01, 0.03, 0.05, 0.1, 0.2],
		"svr__gamma": ["scale", "auto", 0.01, 0.05, 0.1, 0.5, 1.0],
	}

	search = GridSearchCV(
		estimator=base_model,
		param_grid=param_grid,
		cv=LeaveOneOut(),
		scoring="neg_mean_absolute_error",
		n_jobs=-1,
	)
	search.fit(x, y)

	best_model = cast(Pipeline, search.best_estimator_)
	best_params = cast(dict[str, object], search.best_params_)
	best_mae = -float(search.best_score_)
	return best_model, best_params, best_mae


def main() -> None:
	parser = argparse.ArgumentParser(description="使用 HSV/RGB 数据训练 SVR 回归模型预测 pH")
	parser.add_argument("--data", type=str, default="training_data.dat", help="数据文件路径")
	parser.add_argument(
		"--feature",
		type=str,
		choices=["hsv", "rgb"],
		default="hsv",
		help="训练特征类型：hsv 或 rgb",
	)
	parser.add_argument("--c", type=float, default=10.0, help="SVR 的 C 参数")
	parser.add_argument("--epsilon", type=float, default=0.05, help="SVR 的 epsilon 参数")
	parser.add_argument("--gamma", type=str, default="scale", help="SVR 的 gamma 参数")
	parser.add_argument("--auto-tune", action="store_true", help="启用 GridSearchCV 自动调参")
	parser.add_argument("--save", type=str, default="", help="可选：保存模型路径（.pkl）")
	args = parser.parse_args()

	feature = cast(Literal["hsv", "rgb"], args.feature)
	h, s, v, r, g, b, y = parse_data(args.data)
	x = build_features(feature, h, s, v, r, g, b)
	print(f"特征模式: {feature.upper()}")
	print(f"样本数: {len(y)}")

	if args.auto_tune:
		model, best_params, best_mae = auto_tune_model(x, y)
		print("自动调参完成（目标: 最小 MAE）")
		print(f"最佳参数: {best_params}")
		print(f"网格搜索最优 MAE: {best_mae:.4f}")
	else:
		gamma = parse_gamma(args.gamma)
		model = build_model(c=args.c, epsilon=args.epsilon, gamma=gamma)

	# 小样本场景下用留一法评估
	loo = LeaveOneOut()
	y_cv_pred = cross_val_predict(model, x, y, cv=loo)

	mae = mean_absolute_error(y, y_cv_pred)
	rmse = np.sqrt(mean_squared_error(y, y_cv_pred))
	r2 = r2_score(y, y_cv_pred)

	print("留一法交叉验证结果:")
	print(f"MAE : {mae:.4f}")
	print(f"RMSE: {rmse:.4f}")
	print(f"R²  : {r2:.4f}")

	model.fit(x, y)
	train_pred = model.predict(x)
	print("\n训练集拟合示例(真实值 -> 预测值):")
	for yt, yp in zip(y, train_pred):
		print(f"{yt:.2f} -> {yp:.3f}")

	if args.save:
		with open(args.save, "wb") as f:
			pickle.dump(model, f)
		print(f"\n模型已保存到: {args.save}")


if __name__ == "__main__":
	main()
