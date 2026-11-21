"""Графики Matplotlib для notebook. Функции возвращают Figure, не создавая файлов."""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .data import COLOR_LABELS, LABELS

COLORS = {"red": "#a43e54", "white": "#b58b2d"}
BLUE = "#326f9c"


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.titlesize": 15, "axes.labelsize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110, "savefig.dpi": 150,
        "figure.constrained_layout.use": True,
        "axes.axisbelow": True,
    })


def distributions(frame):
    fig, axes = plt.subplots(5, 3, figsize=(11, 18))
    for column, ax in zip(frame.columns, axes.flat):
        if column == "color":
            counts = frame.color.value_counts().reindex(["red", "white"])
            ax.bar([COLOR_LABELS[c] for c in counts.index], counts, color=[COLORS[c] for c in counts.index])
        elif column == "quality":
            counts = frame.quality.value_counts().sort_index()
            ax.bar(counts.index, counts, color=BLUE)
            ax.set_xticks(counts.index)
        else:
            ax.hist(frame[column], bins=30, color=BLUE, edgecolor="white", linewidth=0.5)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_title(LABELS[column])
        ax.set_ylabel("Количество вин")
        ax.grid(axis="y", alpha=0.2)
    for ax in list(axes.flat)[len(frame.columns):]:
        ax.set_visible(False)
    fig.suptitle("Распределения всех 13 столбцов — крайние значения сохранены", fontsize=19)
    return fig


def target_correlations(correlations):
    ordered = correlations.loc[correlations["Все вина"].abs().sort_values().index]
    fig, ax = plt.subplots(figsize=(11, 8))
    positions = np.arange(len(ordered))
    for offset, (label, paint) in zip((-0.24, 0, 0.24), (("Все вина", BLUE), ("Красное", COLORS["red"]), ("Белое", COLORS["white"]))):
        ax.barh(positions + offset, ordered[label], height=0.23, label=label, color=paint)
    ax.set_yticks(positions, [LABELS[name] for name in ordered.index])
    ax.axvline(0, color="#526171", linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Корреляция признака с quality")
    ax.set_title("Связи с оценкой: все вина и отдельно по цветам")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    return fig


def scatter_pairs(frame):
    pairs = [("alcohol", "density"), ("residual sugar", "density"),
             ("free sulfur dioxide", "total sulfur dioxide")]
    fig, axes = plt.subplots(3, 1, figsize=(11, 15))
    for ax, (x, y) in zip(axes, pairs):
        for color in ("red", "white"):
            subset = frame[frame.color == color]
            ax.scatter(subset[x], subset[y], s=9, alpha=0.3, color=COLORS[color], label=COLOR_LABELS[color])
        ax.set_xlabel(LABELS[x])
        ax.set_ylabel(LABELS[y])
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.2)
        ax.legend()
    return fig


def coefficients(values):
    values = values.drop("color_white").sort_values(key=lambda series: series.abs())
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh([LABELS[name] for name in values.index], values,
            color=[BLUE if value >= 0 else COLORS["red"] for value in values])
    ax.axvline(0, color="#526171", linewidth=1)
    ax.set_xlabel("Изменение прогноза при +1 стандартном отклонении train, баллы")
    ax.set_title("Коэффициенты обычной регрессии")
    ax.grid(axis="x", alpha=0.2)
    return fig


def importances(frame):
    ordered = frame.sort_values("forest")
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(positions - 0.19, ordered.forest * 100, height=0.36, label="Random Forest", color=BLUE)
    ax.barh(positions + 0.19, ordered.boosting * 100, height=0.36, label="Gradient Boosting", color=COLORS["white"])
    ax.set_yticks(positions, [LABELS[name] for name in ordered.index])
    ax.set_xlabel("Доля уменьшения ошибки на обучающих разделениях, %")
    ax.set_title("Встроенные важности признаков")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    return fig


def error_distribution(errors):
    edge = max(1, np.ceil(errors.abs().max() * 4) / 4)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(errors, bins=np.arange(-edge, edge + 0.5, 0.25), color=BLUE, edgecolor="white")
    ax.axvline(0, color=COLORS["red"], linestyle="--", label="Точный прогноз")
    ax.set_xlabel("Ошибка = прогноз − настоящая оценка, баллы")
    ax.set_ylabel("Количество вин")
    ax.set_title("Распределение ошибок лучшей модели")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    return fig


def actual_vs_predicted(predictions, key):
    fig, ax = plt.subplots(figsize=(11, 7))
    for color in ("red", "white"):
        subset = predictions[predictions.color == color]
        ax.scatter(subset.quality, subset[f"prediction_{key}"], s=20, alpha=0.35,
                   color=COLORS[color], label=COLOR_LABELS[color])
    ax.plot([3, 9], [3, 9], "--", color="#526171", label="Точный прогноз")
    ax.set(xlim=(2.8, 9.2), ylim=(2.8, 9.2), xticks=range(3, 10), yticks=range(3, 10),
           xlabel="Настоящая оценка quality", ylabel="Предсказанная оценка quality",
           title="Одна точка — одно тестовое вино; точки могут перекрываться")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def errors_by_quality(frame):
    values = frame[frame.n_test > 0]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(values.quality - 0.18, values.mae, width=0.35, label="MAE", color=BLUE)
    ax.bar(values.quality + 0.18, values.rmse, width=0.35, label="RMSE", color=COLORS["white"])
    ax.set_xticks(values.quality, [f"{int(row.quality)}\nn={int(row.n_test)}" for row in values.itertuples()])
    ax.set(xlabel="Настоящая оценка и число тестовых строк", ylabel="Ошибка, баллы",
           title="Точность для разных quality")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    return fig
