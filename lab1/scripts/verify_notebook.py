"""Полный запуск notebook и проверка сохранности данных и результатов."""

import hashlib
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    reference = json.loads((RESULTS / "reference.json").read_text())
    names = ["metrics", "predictions", "coefficients", "feature_importance",
             "error_by_quality", "error_by_color", "largest_errors"]
    previous = {name: pd.read_csv(RESULTS / f"{name}.csv", sep=";") for name in names}
    history = (RESULTS / "experiments.csv").read_bytes()
    notebook = nbformat.read(ROOT / "lab01.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    print("Выполняется notebook из чистого ядра...", flush=True)
    NotebookClient(notebook, timeout=180, kernel_name="ml-lab",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.validate(notebook)
    code_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()
    ]
    assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1))
    figures = 0
    for cell in code_cells:
        for output in cell.outputs:
            assert output.output_type != "error", output
            assert not {"text/html", "application/javascript", "image/svg+xml"} & set(output.get("data", {}))
            assert not (output.output_type == "stream" and output.name == "stderr"), output
            figures += "image/png" in output.get("data", {})
    assert len(code_cells) == 20 and figures == 8

    for relative, digest in reference["data_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    assert (RESULTS / "experiments.csv").read_bytes() == history
    for name, before in previous.items():
        after = pd.read_csv(RESULTS / f"{name}.csv", sep=";")
        if name == "feature_importance":
            before = before.set_index("feature").sort_index()
            after = after.set_index("feature").sort_index()
        pd.testing.assert_frame_equal(after, before, check_exact=False, rtol=0, atol=1e-10)

    metrics = pd.read_csv(RESULTS / "metrics.csv", sep=";", index_col="model")
    expected = pd.DataFrame.from_dict(reference["metrics"], orient="index")
    np.testing.assert_allclose(metrics.loc[expected.index, expected.columns], expected, rtol=0, atol=1e-10)
    config = json.loads((RESULTS / "config.json").read_text())
    assert (config["n_train"], config["n_test"], config["group_overlap"]) == (5183, 1314, 0)
    assert config["random_state"] == 42 and config["ridge_alpha"] == 750
    assert config["best_model"] == "forest"

    by_quality = pd.read_csv(RESULTS / "error_by_quality.csv", sep=";")
    assert by_quality.n_test.sum() == 1314
    quality_nine = by_quality.loc[by_quality.quality == 9].iloc[0]
    assert quality_nine.n_test == 0 and pd.isna(quality_nine.mae)
    nbformat.write(notebook, ROOT / "lab01.ipynb")
    print(f"PASS: {len(code_cells)} ячеек, {figures} рисунков, ошибок и предупреждений в выводе нет.")
    print("PASS: исходные данные, train/test и история экспериментов сохранены.")
    print("PASS: все прогнозы, метрики, коэффициенты, важности и таблицы ошибок совпали с прежними.")
    print(metrics[["rmse", "mae"]].to_string())


if __name__ == "__main__":
    main()
