"""Сохранение таблиц и настроек; обучение и анализ находятся в notebook."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn

from .data import make_group_ids


def save_results(folder, *, tables, models, train, test, model_labels, random_state):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", sep=";",
                     index=name in {"metrics", "feature_importance"})

    coefficient_rows = []
    for key in ("ordinary", "selected"):
        model = models[key]
        values = pd.Series(model.named_steps["regression"].coef_,
                           index=model[:-1].get_feature_names_out())
        values.loc["intercept"] = model.named_steps["regression"].intercept_
        coefficient_rows.extend(
            {"model": key, "feature": name, "coefficient": float(value)}
            for name, value in values.items()
        )
    pd.DataFrame(coefficient_rows).to_csv(folder / "coefficients.csv", sep=";", index=False)

    root = folder.parent
    reference = json.loads((folder / "reference.json").read_text())
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in reference["data_sha256"]}
    groups = make_group_ids(pd.concat([train, test], ignore_index=True))
    train_groups, test_groups = groups.iloc[:len(train)], groups.iloc[len(train):]
    parameter_names = {
        "forest": ["n_estimators", "criterion", "max_depth", "min_samples_split",
                   "min_samples_leaf", "max_features", "bootstrap", "random_state", "n_jobs"],
        "boosting": ["loss", "n_estimators", "learning_rate", "max_depth", "min_samples_split",
                     "min_samples_leaf", "subsample", "max_features", "init",
                     "n_iter_no_change", "random_state"],
    }
    parameters = {
        key: {name: models[key].named_steps["regression"].get_params()[name] for name in names}
        for key, names in parameter_names.items()
    }
    config = {
        "n_train": len(train), "n_test": len(test),
        "n_train_groups": int(train_groups.nunique()), "n_test_groups": int(test_groups.nunique()),
        "group_overlap": len(set(train_groups) & set(test_groups)),
        "random_state": random_state, "test_size_groups": 0.2,
        "baseline_constant": float(train.quality.mean()),
        "best_model": tables["metrics"].drop(index="baseline").rmse.idxmin(),
        "model_labels": model_labels,
        "forest_params": parameters["forest"], "boosting_params": parameters["boosting"],
        "ridge_alpha": models["selected"].named_steps["regression"].alpha,
        "log1p_features": ["residual sugar", "free sulfur dioxide", "total sulfur dioxide"],
        "tuning_on_test": False, "test_previously_inspected": True, "dataset_modified": False,
        "predictions_clipped_or_rounded": False,
        "error_definition": "prediction - actual quality", "data_sha256": hashes,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    (folder / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
