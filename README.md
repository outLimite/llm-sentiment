# Sentiment Analysis with PEFT

Проект для анализа тональности с использованием методов эффективной тонкой настройки.

## Структура проекта

```
sentiment-analysis-peft/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Загрузка и подготовка данных
│   │   └── preprocessing.py    # Препроцессинг и токенизация
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lora.py            # LoRA implementation
│   │   ├── dora.py            # DoRA implementation
│   │   └── qlora.py           # QLoRA implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Основной тренировочный цикл
│   │   ├── eval.py            # Функции оценки
│   │   └── configs.py         # Конфигурации тренировки
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py         # Вспомогательные функции
│   │   └── postprocessing.py  # Постпроцессинг ответов
│   └── inference.py           # Инференс и демо
├── configs/
│   ├── base.yaml              # Базовые настройки
│   ├── lora.yaml              # Конфиг LoRA
│   ├── dora.yaml              # Конфиг DoRA
│   └── qlora.yaml             # Конфиг QLoRA
├── scripts/
│   ├── train.py               # Основной скрипт обучения
│   ├── evaluate.py            # Оценка моделей
│   └── demo.py               # Демо с интерфейсом
├── requirements.txt
├── setup.py
├── .gitattributes
└── README.md
```