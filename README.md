# LLM IoT Intrusion Detection

Проект выполнен в рамках зачета по курсу «Безопасность инфраструктурных технологий».

## Тема

Проверка возможности использования российских LLM для генерации Python-приложений анализа сетевого трафика.

В работе сравниваются:
- ручная baseline-реализация;
- код, сгенерированный YandexGPT;
- код, сгенерированный GigaChat.

## Датасет

Используется датасет IoT Intrusion Detection Dataset с Kaggle:

https://www.kaggle.com/datasets/subhajournal/iotintrusion/data

CSV-файл датасета не включен в репозиторий из-за большого размера.

Для запуска необходимо скачать датасет и поместить файл `IoT_Intrusion.csv` в папку:

```text
data/raw/
```

## Структура проекта

```text
llm-iot-intrusion-detection/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── README.md
│   └── raw/
│       └── .gitkeep
│
├── prompts/
│   ├── prompt_ru.md
│   ├── prompt_en.md
│   ├── yandexgpt_response.md
│   ├── yandexgpt_followup_1_response.md
│   ├── yandexgpt_followup_2_response.md
│   ├── yandexgpt_followup_3_response.md
│   ├── gigachat_response.md
│   ├── gigachat_followup_1_response.md
│   └── gigachat_followup_2_response.md
│
├── src/
│   ├── manual_baseline.py
│   ├── yandexgpt_app.py
│   └── gigachat_app.py
│
├── results/
│   ├── metrics_manual.csv
│   ├── yandexgpt_metrics.txt
│   ├── gigachat_metric_2s.txt
│   ├── comparison_table.md
│   ├── yandexgpt_notes.md
│   └── gigachat_notes.md
│
└── report/
    └── report.md
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск

Ручная baseline-модель:

```bash
python src/manual_baseline.py
```

YandexGPT-версия:

```bash
python src/yandexgpt_app.py
```

GigaChat-версия:

```bash
python src/gigachat_app.py
```

## Краткие результаты

| Подход | Лучшая модель | F1 macro |
|---|---|---:|
| YandexGPT | Decision Tree | 0.8399 |
| Ручная реализация | Decision Tree | 0.8294 |
| GigaChat | Decision Tree | 0.8290 |

## Вывод

Обе LLM смогли сгенерировать рабочий ML-пайплайн только после уточняющих промптов. Основные ошибки были связаны с именем файла, выбором целевого столбца и логикой выбора лучшей модели.

LLM можно использовать как инструмент ускорения разработки, но результат требует обязательной проверки человеком.
