import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


DATA_DIR = "data/raw"
RESULTS_DIR = "results"
MODELS_DIR = "models"
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def find_csv_file(data_dir: str) -> str:
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("В папке data/raw/ не найден CSV-файл.")
    return os.path.join(data_dir, csv_files[0])


def detect_target_column(df: pd.DataFrame) -> str:
    candidates = [
        "label", "Label",
        "class", "Class",
        "attack", "Attack",
        "Attack_type", "attack_type",
        "Category", "category"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    print("Не удалось автоматически определить целевой столбец.")
    print("Список столбцов:")
    for col in df.columns:
        print("-", col)

    target = input("Введите имя целевого столбца: ").strip()

    if target not in df.columns:
        raise ValueError(f"Столбец {target} не найден в датасете.")

    return target


def main():
    file_path = find_csv_file(DATA_DIR)
    print(f"Загружается датасет: {file_path}")

    df = pd.read_csv(file_path)

    print("\n=== Первичный анализ данных ===")
    print(f"Размер датасета: {df.shape}")
    print("\nПервые строки:")
    print(df.head())

    target_col = detect_target_column(df)
    print(f"\nЦелевой столбец: {target_col}")

    print("\nРаспределение классов:")
    print(df[target_col].value_counts())

    # Удаляем полностью пустые строки
    df = df.dropna(how="all")

    # Разделяем признаки и целевую переменную
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Кодируем целевую переменную
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Определяем числовые и категориальные признаки
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print("\nЧисловые признаки:", len(numeric_features))
    print("Категориальные признаки:", len(categorical_features))

    # Предобработка
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    results = []
    best_model_name = None
    best_f1 = -1
    best_pipeline = None

    for model_name, model in models.items():
        print(f"\n=== Обучение модели: {model_name} ===")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision macro: {precision:.4f}")
        print(f"Recall macro: {recall:.4f}")
        print(f"F1 macro: {f1:.4f}")

        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_.astype(str),
            zero_division=0
        )

        cm = confusion_matrix(y_test, y_pred)

        with open(f"{RESULTS_DIR}/{model_name.replace(' ', '_').lower()}_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        pd.DataFrame(cm).to_csv(
            f"{RESULTS_DIR}/{model_name.replace(' ', '_').lower()}_confusion_matrix.csv",
            index=False
        )

        results.append({
            "approach": "manual",
            "model": model_name,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name
            best_pipeline = pipeline

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/metrics_manual.csv", index=False)

    joblib.dump(best_pipeline, f"{MODELS_DIR}/manual_best_model.pkl")

    with open(f"{RESULTS_DIR}/manual_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "best_f1_macro": best_f1
            },
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n=== Итог ручной реализации ===")
    print(results_df)
    print(f"\nЛучшая модель: {best_model_name}, F1 macro = {best_f1:.4f}")


if __name__ == "__main__":
    main()