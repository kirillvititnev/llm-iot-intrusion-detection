import os
import pandas as pd

DATA_DIR = "data/raw"

csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("В папке data/raw/ не найден CSV-файл.")

file_path = os.path.join(DATA_DIR, csv_files[0])
print(f"Загружается файл: {file_path}")

df = pd.read_csv(file_path)

print("\nРазмер датасета:")
print(df.shape)

print("\nПервые 5 строк:")
print(df.head())

print("\nСтолбцы:")
for col in df.columns:
    print(col)

print("\nТипы данных:")
print(df.dtypes)

print("\nКоличество пропусков:")
print(df.isnull().sum())