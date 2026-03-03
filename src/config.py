import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
EVAL_DIR = OUTPUTS_DIR / "evaluation"

for d in [DATA_DIR, SRC_DIR, MODELS_DIR, EDA_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

START_DATE = "2025-03-03"
END_DATE = "2026-03-02"

ZONES = [
    {"id": 1, "name": "Центр-Кремль", "type": "center"},
    {"id": 2, "name": "Центр-Арбат", "type": "center"},
    {"id": 3, "name": "Москва-Сити", "type": "business"},
    {"id": 4, "name": "Тверская", "type": "business"},
    {"id": 5, "name": "Хамовники", "type": "residential"},
    {"id": 6, "name": "Замоскворечье", "type": "residential"},
    {"id": 7, "name": "Таганка", "type": "residential"},
    {"id": 8, "name": "Пресня", "type": "residential"},
    {"id": 9, "name": "Сокольники", "type": "residential"},
    {"id": 10, "name": "Марьино", "type": "suburb"},
    {"id": 11, "name": "Бутово", "type": "suburb"},
    {"id": 12, "name": "Митино", "type": "suburb"},
    {"id": 13, "name": "Жулебино", "type": "suburb"},
    {"id": 14, "name": "Солнцево", "type": "suburb"},
    {"id": 15, "name": "Шереметьево", "type": "airport"},
    {"id": 16, "name": "Домодедово", "type": "airport"},
    {"id": 17, "name": "Внуково", "type": "airport"},
    {"id": 18, "name": "Ленинградский вокзал", "type": "station"},
    {"id": 19, "name": "Казанский вокзал", "type": "station"},
    {"id": 20, "name": "Курский вокзал", "type": "station"},
]

HOLIDAYS = pd.to_datetime([
    "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
    "2025-02-23", "2025-03-08", "2025-05-01", "2025-05-09", "2025-06-12", "2025-11-04", "2025-12-31",
    "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
    "2026-02-23", "2026-03-08"
])
