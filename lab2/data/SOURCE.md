# Источник данных

- Датасет: [yasserh/Titanic Dataset на Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset), прямо указан в ТЗ лабы №2.
- Способ получения: публичный endpoint Kaggle `https://www.kaggle.com/api/v1/datasets/download/yasserh/titanic-dataset`, загрузка 31.10.2026.
- Содержимое архива: один файл `Titanic-Dataset.csv`; файл распакован без преобразований.
- Лицензия, указанная в метаданных Kaggle на момент загрузки: CC0-1.0.
- Контрольная сумма SHA-256 CSV: `7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f`.
- Контрольная сумма SHA-256 скачанного ZIP: `723c095ae74d4261028e19e30fa335a94f5f4bce77716e4bf2365abd128dbb22`.

Проверка целостности при подготовке: 891 строк данных, 12 столбцов, все строки одинаковой ширины; `Survived` содержит значения `0` и `1`. Это только проверка загруженного файла, первичный анализ для лабы ещё не проводился.

Столбцы в исходном порядке: `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.
