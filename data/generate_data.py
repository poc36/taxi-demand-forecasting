import pandas as pd
import numpy as np
from datetime import timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import START_DATE, END_DATE, ZONES, HOLIDAYS, DATA_DIR

np.random.seed(42)

# Realistic hourly demand profiles (fraction of base demand)
PROFILES = {
    "center": [
        0.10, 0.06, 0.04, 0.03, 0.04, 0.08,  # 0-5
        0.20, 0.55, 0.85, 0.90, 0.80, 0.75,  # 6-11
        0.70, 0.65, 0.60, 0.65, 0.75, 0.95,  # 12-17
        1.00, 0.90, 0.70, 0.50, 0.30, 0.15,  # 18-23
    ],
    "business": [
        0.05, 0.03, 0.02, 0.02, 0.03, 0.10,
        0.30, 0.70, 1.00, 0.95, 0.80, 0.70,
        0.60, 0.55, 0.50, 0.55, 0.70, 0.90,
        0.85, 0.60, 0.35, 0.20, 0.10, 0.06,
    ],
    "residential": [
        0.08, 0.05, 0.03, 0.03, 0.05, 0.12,
        0.35, 0.80, 1.00, 0.70, 0.50, 0.45,
        0.40, 0.40, 0.45, 0.55, 0.70, 0.90,
        0.95, 0.75, 0.50, 0.30, 0.18, 0.10,
    ],
    "suburb": [
        0.06, 0.04, 0.03, 0.02, 0.04, 0.15,
        0.50, 0.90, 1.00, 0.60, 0.35, 0.30,
        0.25, 0.25, 0.30, 0.40, 0.60, 0.85,
        0.95, 0.70, 0.40, 0.22, 0.12, 0.08,
    ],
    "airport": [
        0.30, 0.25, 0.20, 0.25, 0.35, 0.55,
        0.70, 0.85, 0.90, 0.85, 0.80, 0.75,
        0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
        1.00, 0.90, 0.75, 0.60, 0.45, 0.35,
    ],
    "station": [
        0.15, 0.10, 0.08, 0.10, 0.15, 0.30,
        0.55, 0.85, 1.00, 0.90, 0.75, 0.65,
        0.60, 0.65, 0.70, 0.75, 0.85, 0.95,
        0.90, 0.75, 0.55, 0.35, 0.25, 0.18,
    ],
}

# Base demand per zone (fixed, not random)
BASE_DEMANDS = {
    1: 90, 2: 85, 3: 50, 4: 40, 5: 25, 6: 20, 7: 18, 8: 30,
    9: 15, 10: 12, 11: 10, 12: 8, 13: 7, 14: 6, 15: 60, 16: 55,
    17: 20, 18: 55, 19: 60, 20: 45,
}


def generate_synthetic_data():
    print("Generating synthetic demand data (v2 — low noise)...")
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='h')
    n = len(dates)

    # Weather — smooth, deterministic + small noise
    day_frac = dates.dayofyear / 365.0
    temp = 5 - 18 * np.cos(2 * np.pi * day_frac) + np.random.normal(0, 1.5, n)
    precip_prob = 0.15 + 0.10 * np.sin(2 * np.pi * day_frac)  # more rain in autumn
    precip_mask = np.random.rand(n) < precip_prob
    precip = np.zeros(n)
    precip[precip_mask] = np.random.exponential(2.0, precip_mask.sum())
    wind = 4 + 2 * np.sin(2 * np.pi * day_frac) + np.random.normal(0, 1, n)
    wind = np.clip(wind, 0, None)

    # Holiday mask for all dates
    holiday_dates = set(h.date() for h in HOLIDAYS)
    is_holiday = np.array([d.date() in holiday_dates for d in dates], dtype=int)

    records = []

    for zone in ZONES:
        zid = zone["id"]
        ztype = zone["type"]
        base = BASE_DEMANDS[zid]
        profile = np.array(PROFILES[ztype])

        # Hourly profile for every timestamp
        hourly = profile[dates.hour]

        # Day-of-week effect: weekends different
        dow = dates.dayofweek
        dow_coeff = np.ones(n)
        if ztype in ["center", "station", "airport"]:
            dow_coeff[dow >= 5] = 1.15  # busier on weekends
        elif ztype in ["business"]:
            dow_coeff[dow >= 5] = 0.55  # quiet on weekends
        else:
            dow_coeff[dow >= 5] = 1.05

        # Monthly seasonality: summer slightly less, December/January higher
        month_coeff = 1.0 + 0.08 * np.sin(2 * np.pi * (dates.month - 1) / 12)

        # Holiday effect
        hol_coeff = np.ones(n)
        hol_coeff[is_holiday == 1] = 1.25 if ztype in ["center", "airport", "station"] else 0.85

        # Weather effect: rain increases taxi demand, extreme cold too
        weather_coeff = np.ones(n)
        weather_coeff[precip > 0] += 0.08
        weather_coeff[precip > 5] += 0.12
        weather_coeff[temp < -10] += 0.10

        # Final demand: deterministic + small Gaussian noise (~3% of value)
        demand = base * hourly * dow_coeff * month_coeff * hol_coeff * weather_coeff
        noise = np.random.normal(0, demand * 0.03)  # 3% noise
        demand = np.clip(np.round(demand + noise), 0, None).astype(int)

        zone_df = pd.DataFrame({
            "datetime": dates,
            "zone_id": zid,
            "zone_name": zone["name"],
            "zone_type": ztype,
            "temperature": np.round(temp, 1),
            "precipitation": np.round(precip, 1),
            "wind_speed": np.round(wind, 1),
            "is_holiday": is_holiday,
            "demand": demand,
        })
        records.append(zone_df)

    df = pd.concat(records, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "taxi_demand.csv", index=False)
    print(f"Generated {len(df):,} rows → {DATA_DIR / 'taxi_demand.csv'}")
    print(f"  Date range: {dates[0]} → {dates[-1]}")
    print(f"  Avg demand: {df['demand'].mean():.1f}")


if __name__ == "__main__":
    generate_synthetic_data()
