"""Проверки данных, группы повторов и диагностика отдельных вин."""

import json

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]
FEATURE_COLUMNS = [*NUMERIC_FEATURES, "color"]
LABELS = {
    "fixed acidity": "Нелетучая кислотность", "volatile acidity": "Летучая кислотность",
    "citric acid": "Лимонная кислота", "residual sugar": "Остаточный сахар",
    "chlorides": "Хлориды", "free sulfur dioxide": "Свободный диоксид серы",
    "total sulfur dioxide": "Общий диоксид серы", "density": "Плотность",
    "pH": "pH", "sulphates": "Сульфаты", "alcohol": "Крепость",
    "quality": "Оценка качества", "color": "Цвет вина", "color_white": "Цвет: белое вино",
}
COLOR_LABELS = {"red": "Красное", "white": "Белое"}


def validate_data(frame: pd.DataFrame) -> None:
    required = [*FEATURE_COLUMNS, "quality"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"В данных отсутствуют столбцы: {sorted(missing)}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Таблица пуста или содержит пропуски.")
    numeric = frame[[*NUMERIC_FEATURES, "quality"]]
    if not all(pd.api.types.is_numeric_dtype(numeric[name]) for name in numeric):
        raise ValueError("Химические показатели и quality должны быть числовыми.")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Обнаружены бесконечные числовые значения.")
    if not frame.color.isin(["red", "white"]).all():
        raise ValueError("Ожидаются цвета red и white.")


def make_group_ids(frame: pd.DataFrame) -> pd.Series:
    """Одинаковые исходные показатели и цвет — одна группа, без использования quality."""
    return frame.groupby(FEATURE_COLUMNS, sort=True, dropna=False).ngroup()


def diagnose_largest_errors(train: pd.DataFrame, predictions: pd.DataFrame, n=10) -> pd.DataFrame:
    """Разные наборы измерений с крупными ошибками и пять соседей train того же цвета.

    Расстояния считаются без quality, масштабы и процентили — только по train.
    """
    largest = predictions.sort_values(["absolute_error", "test_row_index"], ascending=[False, True])
    largest = largest.drop_duplicates(subset=FEATURE_COLUMNS).head(n).copy()
    scales = train[NUMERIC_FEATURES].std(ddof=0).replace(0, 1).to_numpy()
    for index, row in largest.iterrows():
        same_color = train[train.color == row.color]
        distances = np.linalg.norm((same_color[NUMERIC_FEATURES].to_numpy()
                                   - row[NUMERIC_FEATURES].to_numpy(dtype=float)) / scales, axis=1)
        neighbors = same_color.iloc[np.argsort(distances, kind="stable")[:5]]
        percentiles = {name: float((same_color[name] <= row[name]).mean() * 100) for name in NUMERIC_FEATURES}
        unusual = [name for name in NUMERIC_FEATURES if percentiles[name] <= 1 or percentiles[name] >= 99]
        unusual.sort(key=lambda name: min(percentiles[name], 100 - percentiles[name]))
        largest.loc[index, "neighbor_qualities"] = ", ".join(str(int(q)) for q in neighbors.quality)
        largest.loc[index, "neighbor_mean_quality"] = float(neighbors.quality.mean())
        largest.loc[index, "n_train_same_quality"] = int((train.quality == row.quality).sum())
        largest.loc[index, "unusual_features"] = json.dumps({
            name: {"value": float(row[name]), "percentile": percentiles[name]} for name in unusual[:3]
        }, ensure_ascii=False)
    return largest
