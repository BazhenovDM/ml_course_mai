# Происхождение датасета Palmer Penguins

- Файл: [`penguins.csv`](penguins.csv), исходный CSV без изменений.
- Официальный проект: [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/); описание полей — [документация `penguins`](https://allisonhorst.github.io/palmerpenguins/reference/penguins.html).
- Источник загрузки: [`inst/extdata/penguins.csv`](https://github.com/allisonhorst/palmerpenguins/blob/8957207b78d6ccd1b4654a9dd9c9041b657478ab/inst/extdata/penguins.csv) на коммите `8957207b78d6ccd1b4654a9dd9c9041b657478ab`; загружен 31.10.2026.
- SHA-256 исходного CSV: `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93`.
- Лицензия данных: [CC0](https://allisonhorst.github.io/palmerpenguins/#license). Исходные наблюдения собрали Kristen Gorman и Palmer Station LTER; подробные ссылки на первичные наборы приведены в [документации](https://allisonhorst.github.io/palmerpenguins/reference/penguins.html#source).

## Проверка при подготовке

В файле 344 строки наблюдений и 8 столбцов: `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`, `year`. Значение `NA` обозначает пропуск. Пропуски: `sex` — 11; каждый из четырёх числовых признаков — 2; в остальных столбцах — 0. Для предварительно выбранных пяти переменных (`species`, `island`, `sex`, `bill_length_mm`, `flipper_length_mm`) есть 333 полные строки. Это только инвентаризация входных данных; подготовка признаков и обучение сети ещё не выполнялись.

Исходный файл следует оставлять неизменным. Правила фильтрации и дискретизации должны быть записаны в будущем notebook.
