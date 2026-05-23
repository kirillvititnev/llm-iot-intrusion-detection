
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump

# Загрузка данных
def load_data(file_path):
    """
    Загружает данные из CSV-файла и определяет целевой столбец.
    """
    df = pd.read_csv(file_path)
    target_column = None
    for column in df.columns:
        if 'class' in column.lower() or 'label' in column.lower():
            target_column = column
            break
    if target_column is None:
        print("Не удалось автоматически определить целевой столбец. Пожалуйста, укажите его имя.")
        target_column = input("Введите имя целевого столбца: ")
    return df, target_column

# Первичный анализ данных
def analyze_data(df, target_column):
    """
    Проводит первичный анализ данных.
    """
    print("Размер датасета:", df.shape)
    print("Список признаков:", df.columns)
    print("Количество пропусков:", df.isnull().sum())
    print("Распределение классов:", df[target_column].value_counts())

# Предобработка данных
def preprocess_data(df, target_column):
    """
    Выполняет предобработку данных.
    """
    # Разделение на признаки и целевой столбец
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Обработка пропусков
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

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
        ]
    )

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Применение предобработки
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test

# Обучение моделей
def train_models(X_train, y_train):
    """
    Обучает модели и возвращает их.
    """
    rf = RandomForestClassifier(random_state=42)
    lr = LogisticRegression(random_state=42)
    dt = DecisionTreeClassifier(random_state=42)

    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    return rf, lr, dt

# Оценка моделей
def evaluate_models(X_test, y_test, models):
    """
    Оценивает модели и возвращает метрики.
    """
    metrics = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        confusion = confusion_matrix(y_test, y_pred)

        metrics[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion
        }

    return metrics

# Сохранение результатов
def save_results(metrics, file_path):
    """
    Сохраняет результаты в файл.
    """
    with open(file_path, 'w') as f:
        for model_name, metric in metrics.items():
            f.write(f"{model_name}:\n")
            f.write(f"Accuracy: {metric['accuracy']}\n")
            f.write(f"Precision: {metric['precision']}\n")
            f.write(f"Recall: {metric['recall']}\n")
            f.write(f"F1: {metric['f1']}\n")
            f.write(f"Confusion Matrix: {metric['confusion_matrix']}\n")

# Сохранение лучшей модели
def save_best_model(models, file_path):
    """
    Сохраняет лучшую модель в файл.
    """
    best_model = max(models, key=lambda x: evaluate_models(X_test, y_test, {x: models[x]})['accuracy'])
    dump(models[best_model], file_path)

# Основной код
if __name__ == '__main__':
    # Загрузка данных
    file_path = 'data/raw/IoT_Intrusion_Detection_Dataset.csv'
    df, target_column = load_data(file_path)

    # Первичный анализ данных
    analyze_data(df, target_column)

    # Предобработка данных
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column)

    # Обучение моделей
    rf, lr, dt = train_models(X_train, y_train)

    # Оценка моделей
    models = {
        'Random Forest': rf,
        'Logistic Regression': lr,
        'Decision Tree': dt
    }
    metrics = evaluate_models(X_test, y_test, models)

    # Сохранение результатов
    save_results(metrics, 'results/metrics.txt')

    # Сохранение лучшей модели
    save_best_model(models, 'models/best_model.pkl')

    # Вывод лучшего результата
    best_model_name = max(metrics, key=lambda x: metrics[x]['accuracy'])
    print(f"Лучшая модель: {best_model_name} с точностью {metrics[best_model_name]['accuracy']}")