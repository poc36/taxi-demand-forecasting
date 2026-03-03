import pandas as pd
import numpy as np
from datetime import timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import START_DATE, END_DATE, ZONES, HOLIDAYS, DATA_DIR

def generate_synthetic_data():
    print("Generating synthetic demand data...")
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='H')
    
    records = []
    
    # Base profiles for different zone types
    profiles = {
        "center": np.sin(np.pi * (dates.hour - 8) / 12) + 1.5,
        "business": np.exp(-0.1 * (dates.hour - 14)**2) + 0.5,
        "residential": np.exp(-0.1 * (dates.hour - 8)**2) + np.exp(-0.1 * (dates.hour - 18)**2) + 0.2,
        "suburb": np.exp(-0.1 * (dates.hour - 7)**2) + np.exp(-0.1 * (dates.hour - 19)**2) + 0.1,
        "airport": np.ones(len(dates)) * 0.8 + np.sin(np.pi * dates.hour / 12) * 0.2,
        "station": np.sin(np.pi * (dates.hour - 10) / 12) + 1.0,
    }
    
    # Weather simulation
    temp = 15 - 15 * np.cos(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 3, len(dates))
    precip = np.random.exponential(scale=1.0, size=len(dates)) * (np.random.rand(len(dates)) < 0.2)
    wind = np.random.normal(5, 2, len(dates))
    wind = np.clip(wind, 0, None)
    
    for zone in ZONES:
        base_demand = np.random.uniform(50, 200) if zone["type"] in ["center", "airport", "station"] else np.random.uniform(10, 80)
        profile = profiles[zone["type"]]
        
        # Add day of week effects
        dow_effect = np.ones(len(dates))
        dow_effect[dates.dayofweek >= 5] = 1.2 if zone["type"] in ["center", "residential"] else 0.7
        
        # Add holiday effects
        holiday_effect = np.ones(len(dates))
        for hd in HOLIDAYS:
            mask = (dates.date == hd.date())
            holiday_effect[mask] = 1.3 if zone["type"] in ["center", "airport", "station"] else 0.8
            
        # Add random events
        event_effect = np.ones(len(dates))
        if np.random.rand() > 0.5:
            event_idx = np.random.choice(len(dates), size=10, replace=False)
            event_effect[event_idx] = 2.0
            
        # Weather effects
        weather_effect = np.ones(len(dates))
        weather_effect -= (precip > 0) * 0.1  # Rain reduces base demand slightly
        weather_effect += (precip > 5) * 0.2  # Heavy rain increases taxi demand
        weather_effect += (temp < -10) * 0.15 # Very cold increases demand
        
        # Final demand calculation with noise
        demand = base_demand * profile * dow_effect * holiday_effect * event_effect * weather_effect
        noise = np.random.normal(0, base_demand * 0.1, len(dates))
        demand = np.clip(demand + noise, 0, None)
        
        zone_df = pd.DataFrame({
            "datetime": dates,
            "zone_id": zone["id"],
            "zone_name": zone["name"],
            "zone_type": zone["type"],
            "temperature": temp,
            "precipitation": precip,
            "wind_speed": wind,
            "is_holiday": np.isin(dates.normalize(), HOLIDAYS).astype(int),
            "demand": np.round(demand).astype(int)
        })
        records.append(zone_df)
        
    df = pd.concat(records, ignore_index=True)
    df.to_csv(DATA_DIR / "taxi_demand.csv", index=False)
    print(f"Generated {len(df)} rows. Saved to {DATA_DIR / 'taxi_demand.csv'}")

if __name__ == "__main__":
    generate_synthetic_data()
