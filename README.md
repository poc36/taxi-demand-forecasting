# 🚕 Taxi Demand Forecasting
*[🇷🇺 Читать на русском языке (Russian version below)](#-прогнозирование-спроса-на-такси)*

**Predict hourly taxi demand across 20 city zones** using gradient boosting (LightGBM) with a seasonal baseline comparison. Includes an interactive dark-themed dashboard with historical analysis and 30-day future forecasting.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Dashboard-009688)

---

## 📸 Dashboard

### Heatmap & Metrics
![Dashboard Main](assets/dashboard_main.png)

### Charts: Fact vs Forecast, Error Distribution, Hourly Demand
![Dashboard Charts](assets/dashboard_charts.png)

---

## 📋 Problem Statement

Given historical taxi demand data across **20 city zones** with weather, holidays, and temporal patterns:
- **Predict** the number of orders per zone for each hour
- **Forecast** up to 30 days into the future using rolling predictions
- **Compare** LightGBM vs. seasonal baseline
- **Visualize** everything on an interactive dashboard

## 📈 Results

| Model | MAE | MAPE |
|-------|-----|------|
| **LightGBM** | **1.13** | **8.5%** |
| Seasonal Baseline | 1.81 | 9.1% |

LightGBM outperforms the hourly seasonal baseline, proving the value of the 45 complex features (lags, weather, momentum).

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python data/generate_data.py          # Generate synthetic data (~175k rows)
python src/features.py                # Engineer 45 features
python src/train_lightgbm.py          # Train LightGBM model
python src/evaluate.py                # Compute metrics
python src/forecast_future.py --days 30  # 30-day rolling forecast

# 3. Launch dashboard
python dashboard/app.py
# Open http://localhost:8050
```

## 📊 Feature Engineering (45 features)

| Category | Features |
|----------|----------|
| **Lags** | 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h |
| **Rolling** | Mean/Std over 3h, 6h, 12h, 24h windows |
| **Cyclical** | sin/cos encoding for hour, day-of-week, month |
| **Flags** | Rush hour, night, weekend, holiday |
| **Weather** | Temperature, precipitation, wind speed, bad weather interaction |
| **Zone** | Mean demand per zone, peak hour, demand ratio |

## 🏗️ Project Structure

```
├── data/
│   └── generate_data.py       # Synthetic data generator (seed=42)
├── src/
│   ├── config.py              # Zones, holidays, paths
│   ├── features.py            # Feature engineering pipeline
│   ├── train_lightgbm.py      # Model training with early stopping
│   ├── evaluate.py            # Metrics calculation (MAE, MAPE)
│   └── forecast_future.py     # Rolling 30-day future forecast
├── dashboard/
│   ├── app.py                 # FastAPI backend (5 API endpoints)
│   └── templates/
│       └── index.html         # Interactive dark-themed UI
├── outputs/evaluation/        # Predictions, metrics, forecasts
├── requirements.txt
└── README.md
```

## 🖥️ Dashboard Features

- **Heatmap** — demand intensity per zone at any hour
- **Time slider** — explore demand by hour of day
- **Time series chart** — fact vs prediction with future forecast boundary
- **Error distribution** — histogram of prediction errors
- **Hourly forecast** — average predicted demand by hour
- **Model selector** — switch between LightGBM only or both models
- **Date picker** — grouped by History (with actuals) and Forecast (future)

---

# 🇷🇺 Прогнозирование спроса на такси

**Прогноз почасового спроса на такси в 20 зонах города** с использованием градиентного бустинга (LightGBM) и сравнением с сезонным бейзлайном. Включает интерактивный дашборд в тёмной теме для анализа исторического спроса и прогноза на 30 дней вперёд.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Дашборд-009688)

---

## 📸 Дашборд

### Тепловая карта и метрики
![Dashboard Main](assets/dashboard_main.png)

### Графики: Факт vs Прогноз, распределение ошибок, спрос по часам
![Dashboard Charts](assets/dashboard_charts.png)

---

## 📋 Описание задачи

На основе исторических данных о заказах такси по **20 зонам города**, а также погодных условий, праздников и временных паттернов:
- **Предсказать** количество заказов в каждой зоне на каждый час
- **Сделать прогноз** на 30 дней вперёд с использованием rolling-метода
- **Сравнить** LightGBM с наивным сезонным бейзлайном
- **Визуализировать** результаты на интерактивном дашборде

## 📈 Результаты

| Модель | MAE | MAPE |
|--------|-----|------|
| **LightGBM** | **1.13** | **8.5%** |
| Сезонный Baseline | 1.81 | 9.1% |

LightGBM превосходит почасовой сезонный бейзлайн, доказывая эффективность созданных 45 комплексных признаков (лаги, погода, моментум спроса).

## 🚀 Быстрый старт

```bash
# 1. Загрузите зависимости
pip install -r requirements.txt

# 2. Запустите полный пайплайн
python data/generate_data.py          # Генерация синтетических данных (~175к строк)
python src/features.py                # Создание 45 признаков (Feature Engineering)
python src/train_lightgbm.py          # Обучение модели LightGBM
python src/evaluate.py                # Расчёт метрик (MAE, MAPE)
python src/forecast_future.py --days 30  # Прогноз на 30 дней вперёд

# 3. Запуск дашборда
python dashboard/app.py
# Откройте http://localhost:8050
```

## 📊 Feature Engineering (45 признаков)

| Категория | Признаки |
|-----------|----------|
| **Лаги (Lags)** | 1ч, 2ч, 3ч, 6ч, 12ч, 24ч, 48ч, 168ч |
| **Скользящие (Rolling)** | Среднее и ст. отклонение (Mean/Std) за 3ч, 6ч, 12ч, 24ч |
| **Циклические** | sin/cos кодирование часа, дня недели, месяца |
| **Флаги** | Час пик, ночь, выходной, праздник |
| **Погода** | Температура, осадки, скорость ветра, влияние плохой погоды в час пик |
| **Зоны** | Средний спрос в зоне, час пика в зоне, отношение к среднему спросу |

## 🖥️ Функции дашборда

- **Тепловая карта (Heatmap)** — интенсивность спроса по районам для любого часа
- **Временной слайдер** — переключение часов в рамках выбранных суток
- **График временного ряда** — факт против прогноза (с границей между историей и будущим)
- **Распределение ошибок** — гистограмма, показывающая смещение и точность предсказаний
- **Даты** — удобная группировка дат на "Историю (с фактом)" и "Будущее (Прогноз)"
- **Выбор модели** — наглядное сравнение работы LightGBM и Сезонного бейзлайна

---
*Pet-проект, демонстрирующий полный ML-цикл: от генерации данных и feature engineering до обучения моделей (Time-Series) и создания full-stack дашборда на FastAPI.*
