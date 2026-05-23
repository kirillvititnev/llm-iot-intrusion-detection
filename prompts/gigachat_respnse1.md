Вот готовый к запуску Python-файл, оформленный по всем вашим требованиям. Сохраните его как `src/generated_app.py` и убедитесь, что структура папок выглядит так:

```
project/
├── data/
│   └── raw/
│       └── IoT_Intrusion_Detection_Dataset.csv
├── models/
├── results/
└── src/
    └── generated_app.py
```

### Код приложения

```python
# src/generated_app.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

# --- 1. Загрузка данных ---
def load_data(filepath):
    """Загружает CSV-файл и возвращает DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filepath} не найден.")
    df = pd.read_csv(filepath)
    return df

# --- 2. Определение целевого столбца ---
def find_target_column(df):
    """Автоматически определяет целевой столбец или просит пользователя выбрать."""
    # Предположим, что целевой столбец содержит только 0 и 1 (или больше классов)
    potential_targets = []
    for col in df.columns:
        unique_vals = df[col].nunique()
        if unique_vals <= 10 and df[col].dtype in [int, 'int64', 'object']:
            potential_targets.append(col)
    
    if len(potential_targets) == 1:
        return potential_targets[0]
    elif len(potential_targets) > 1:
        print("Не удалось однозначно определить целевой столбец. Пожалуйста, выберите:")
        for i, col in enumerate(potential_targets):
            print(f"{i+1}. {col}")
        choice = int(input("Введите номер столбца: ")) - 1
        return potential_targets[choice]
    else:
        raise ValueError("Целевой столбец не найден. Проверьте данные.")

# --- 3. Первичный анализ данных ---
def analyze_data(df, target_col):
    """Проводит первичный анализ и выводит статистику."""
    print("\n--- Первичный анализ данных ---")
    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"Признаки: {df.columns.tolist()}")
    print(f"Пропуски:\n{df.isnull().sum()}")
    print(f"Распределение классов:\n{df[target_col].value_counts()}")
    
    # График распределения классов
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=df)
    plt.title("Распределение классов")
    plt.savefig('results/class_distribution.png')
    plt.close()

# --- 4. Предобработка данных ---
def preprocess_data(df, target_col):
    """Выполняет предобработку: обработка пропусков, кодирование, масштабирование."""
    # Удаляем целевой столбец из признаков
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Обработка пропусков: для числовых — медианой, для категориальных — модой
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Разделение на train/test (без утечки данных)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, preprocessor

# --- 5. Обучение моделей ---
def train_models(X_train, y_train):
    """Обучает RandomForest и LogisticRegression/DecisionTree."""
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        # 'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe
        
    return trained_models

# --- 6. Оценка моделей ---
def evaluate_models(trained_models, X_test, y_test):
    """Вычисляет метрики и сохраняет confusion matrix."""
    results = {}
    best_model_name = None
    best_f1 = -1

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'report': classification_report(y_test, y_pred)
        }
        
        # Сохраняем confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f'results/cm_{name}.png')
        plt.close()
        
        # Поиск лучшей модели по F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    return results, best_model_name

# --- 7-8. Сохранение результатов и модели ---
def save_results(results, best_model_name, trained_models):
    """Сохраняет результаты в файлы и лучшую модель."""
    os.makedirs('results', exist_ok=True)
    
    with open('results/metrics.txt', 'w') as f:
        for name, metrics in results.items():
            f.write(f"\n=== {name} ===\n")
            f.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
            f.write(f"Precision (macro): {metrics['precision']:.3f}\n")
            f.write(f"Recall (macro): {metrics['recall']:.3f}\n")
            f.write(f"F1-score (macro): {metrics['f1']:.3f}\n")
            f.write(metrics['report'])
    
    # Сохраняем лучшую модель
    os.makedirs('models', exist_ok=True)
    joblib.dump(trained_models[best_model_name], 'models/best_model.pkl')

# --- Основная функция ---
if __name__ == "__main__":
    # Параметры
    DATA_PATH = 'data/raw/IoT_Intrusion_Detection_Dataset.csv'
    
    # Загрузка данных
    df = load_data(DATA_PATH)
    
    # Определение целевого столбца
    target_col = find_target_column(df)
    
    # Анализ данных
    analyze_data(df, target_col)
    
    # Предобработка
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, target_col)
    
    # Обучение моделей
    trained_models = train_models(X_train, y_train)
    
    # Оценка моделей
    results, best_model_name = evaluate_models(trained_models, X_test, y_test)
    
    # Сохранение результатов и модели
    save_results(results, best_model_name, trained_models)
    
    # Вывод краткого результата в консоль
    best_metrics = results[best_model_name]
    print("\n--- Итоговый результат ---")
    print(f"Лучшая модель: {best_model_name}")
    print(f"Accuracy: {best_metrics['accuracy']:.3f}")
    print(f"F1-score (macro): {best_metrics['f1']:.3f}")
```

### Инструкция по запуску

1. Установите необходимые библиотеки:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```
2. Поместите файл `IoT_Intrusion_Detection_Dataset.csv` в папку `data/raw/`.
3. Запустите приложение:
   ```bash
   python src/generated_app.py
   ```
4. После выполнения в папках `results/` появятся графики и файл с метриками, а лучшая модель будет сохранена в `models/best_model.pkl`.
5. В консоли будет выведено название лучшей модели и её ключевые метрики.

### Особенности реализации

- Код полностью воспроизводим (`random_state=42`).
- Нет утечки данных между train и test.
- Обработка пропусков и категориальных признаков автоматизирована.
- Поддерживается multiclass classification.
- Все этапы сопровождаются комментариями.
- Код оформлен как один исполняемый файл.

Если потребуется доработать визуализацию или добавить новые модели — сообщите!