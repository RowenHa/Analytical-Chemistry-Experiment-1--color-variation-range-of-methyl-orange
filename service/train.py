from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Literal, cast

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


LINE_PATTERN = re.compile(
	r"H=(?P<H>[\d.]+)°\s*,\s*S=(?P<S>[\d.]+)%\s*,\s*V=(?P<V>[\d.]+)%\s*,\s*pH=(?P<pH>[\d.]+)"
)


def parse_data(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
	"""从 color.dat 解析 HSV 和 pH 数据。"""
	rows: list[tuple[float, float, float, float]] = []
	for line in Path(file_path).read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		m = LINE_PATTERN.search(line)
		if m is None:
			continue
		h = float(m.group("H"))
		s = float(m.group("S"))
		v = float(m.group("V"))
		ph = float(m.group("pH"))
		rows.append((h, s, v, ph))

	if not rows:
		raise ValueError("未在数据文件中解析到有效样本")

	arr = np.array(rows, dtype=np.float64)
	h_deg = arr[:, 0]
	s = arr[:, 1]
	v = arr[:, 2]
	y = arr[:, 3]

	# 色相是环形变量，用 sin/cos 编码避免 0° 与 360° 的断点问题
	h_rad = np.deg2rad(h_deg)
	x = np.column_stack([np.sin(h_rad), np.cos(h_rad), s, v])
	return x, y


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
	parser = argparse.ArgumentParser(description="使用 HSV 数据训练 SVR 回归模型预测 pH")
	parser.add_argument("--data", type=str, default="color.dat", help="数据文件路径")
	parser.add_argument("--c", type=float, default=10.0, help="SVR 的 C 参数")
	parser.add_argument("--epsilon", type=float, default=0.05, help="SVR 的 epsilon 参数")
	parser.add_argument("--gamma", type=str, default="scale", help="SVR 的 gamma 参数")
	parser.add_argument("--auto-tune", action="store_true", help="启用 GridSearchCV 自动调参")
	parser.add_argument("--save", type=str, default="", help="可选：保存模型路径（.pkl）")
	args = parser.parse_args()

	x, y = parse_data(args.data)
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

	print(f"样本数: {len(y)}")
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
