import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, ZONES

def engineer_features(df, is_training=False):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['zone_id', 'datetime']).reset_index(drop=True)
    
    # 1. Time based features
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['dayofmonth'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['hour'].isin([8, 9, 18, 19]).astype(int)
    df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
    
    # 2. Cyclical time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 3. Lag features
    lags = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in lags:
        df[f'lag_{lag}h'] = df.groupby('zone_id')['demand'].shift(lag)
        
    # 4. Rolling statistics
    windows = [3, 6, 12, 24]
    for w in windows:
        # Avoid data leakage by shifting first
        shifted = df.groupby('zone_id')['demand'].shift(1)
        df[f'rolling_mean_{w}h'] = shifted.groupby(df['zone_id']).transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'rolling_std_{w}h'] = shifted.groupby(df['zone_id']).transform(lambda x: x.rolling(w, min_periods=1).std())
        
    # 5. Weather interaction features
    df['bad_weather'] = ((df['precipitation'] > 0) | (df['wind_speed'] > 10)).astype(int)
    df['weather_rush_hour'] = df['bad_weather'] * df['is_rush_hour']
    
    # 6. Demand diffs (momentum)
    df['demand_diff_1h'] = df['lag_1h'] - df['lag_2h']
    df['demand_diff_24h'] = df['lag_24h'] - df.groupby('zone_id')['demand'].shift(25)
    
    # 7. Zone features
    stats_file = DATA_DIR / "zone_stats.json"
    
    if is_training:
        zone_means = df.groupby('zone_id')['demand'].mean().to_dict()
        
        zone_peak_hour = df.groupby(['zone_id', 'hour'])['demand'].mean().reset_index()
        zone_peak_hour = zone_peak_hour.loc[zone_peak_hour.groupby('zone_id')['demand'].idxmax()]
        peak_hours = dict(zip(zone_peak_hour['zone_id'], zone_peak_hour['hour']))
        
        # Convert keys to int strings for JSON
        stats = {
            "means": {str(k): v for k, v in zone_means.items()},
            "peaks": {str(k): v for k, v in peak_hours.items()}
        }
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
    else:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        zone_means = {int(k): v for k, v in stats['means'].items()}
        peak_hours = {int(k): v for k, v in stats['peaks'].items()}
        
    df['zone_mean_demand'] = df['zone_id'].map(zone_means)
    df['zone_demand_ratio'] = df['lag_1h'] / (df['zone_mean_demand'] + 1)
    df['zone_peak_hour'] = df['zone_id'].map(peak_hours)
    
    # Drop rows with NaN due to lags (first 168 hours)
    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "taxi_demand.csv")
    print("Engineering features...")
    df_features = engineer_features(df, is_training=True)
    output_path = DATA_DIR / "taxi_demand_features.csv"
    df_features.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}. Shape: {df_features.shape}")
