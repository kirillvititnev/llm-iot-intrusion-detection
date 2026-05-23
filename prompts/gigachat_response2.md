Вот исправленный код. Теперь он автоматически находит первый CSV-файл в папке `data/raw/` и не использует жёстко заданное имя файла.

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
from sklearn.impute import SimpleImputer

# --- 1. Загрузка данных (исправлено: автоматический поиск CSV) ---
def load_data(folder_path):
    """Автоматически находит и загружает первый CSV-файл в указанной папке."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Папка {folder_path} не найдена.")
    
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"В папке {folder_path} не найдено ни одного CSV-файла.")
    
    filepath = os.path.join(folder_path, csv_files[0])
    df = pd.read_csv(filepath)
    print(f"Загружен файл: {csv_files[0]}")
    return df

# --- 2. Определение целевого столбца ---
def find_target_column(df):
    """Автоматически определяет целевой столбец или просит пользователя выбрать."""
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
    
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=df)
    plt.title("Распределение классов")
    plt.savefig('results/class_distribution.png')
    plt.close()

# --- 4. Предобработка данных ---
def preprocess_data(df, target_col):
    """Выполняет предобработку: обработка пропусков, кодирование, масштабирование."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, preprocessor

# --- 5. Обучение моделей ---
def train_models(X_train, y_train, preprocessor):
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
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f'results/cm_{name}.png')
        plt.close()
        
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
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(trained_models[best_model_name], 'models/best_model.pkl')

# --- Основная функция ---
if __name__ == "__main__":
    DATA_FOLDER = 'data/raw/'
    
    try:
        df = load_data(DATA_FOLDER)
        
        target_col = find_target_column(df)
        
        analyze_data(df, target_col)
        
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, target_col)
        
        trained_models = train_models(X_train, y_train, preprocessor)
        
        results, best_model_name = evaluate_models(trained_models, X_test, y_test)
        
        save_results(results, best_model_name, trained_models)
        
        best_metrics = results[best_model_name]
        print("\n--- Итоговый результат ---")
        print(f"Лучшая модель: {best_model_name}")
        print(f"Accuracy: {best_metrics['accuracy']:.3f}")
        print(f"F1-score (macro): {best_metrics['f1']:.3f}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
```

### Краткое объяснение изменений

**Почему возникала ошибка?**
В исходном коде путь к файлу был жёстко задан как `data/raw/IoT_Intrusion_Detection_Dataset.csv`. Если реальное имя файла отличается (например, `IoT_Intrusion.csv`), программа не могла найти файл и выдавала ошибку.

**Что было исправлено?**
Функция `load_data` теперь:
- Принимает на вход только путь к папке (`data/raw/`).
- Автоматически ищет все файлы с расширением `.csv` в этой папке.
- Загружает первый найденный CSV-файл.
- Если файлов нет — выводит понятную ошибку.
- Выводит в консоль имя загруженного файла для контроля.

Вся остальная логика (анализ данных, обучение моделей и т. д.) осталась без изменений.