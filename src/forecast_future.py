import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, MODELS_DIR, EVAL_DIR
from src.features import engineer_features

def generate_future_weather(dates):
    temp = 15 - 15 * np.cos(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 3, len(dates))
    precip = np.random.exponential(scale=1.0, size=len(dates)) * (np.random.rand(len(dates)) < 0.2)
    wind = np.random.normal(5, 2, len(dates))
    wind = np.clip(wind, 0, None)
    return pd.DataFrame({'datetime': dates, 'temperature': temp, 'precipitation': precip, 'wind_speed': wind})

def forecast_future(days=30):
    model_path = MODELS_DIR / "lightgbm_model.pkl"
    if not model_path.exists():
        print("Model not found. Train first.")
        return
        
    model = joblib.load(model_path)
    df = pd.read_csv(DATA_DIR / "taxi_demand.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    last_date = df['datetime'].max()
    
    # We need a buffer of history to compute 168h lags
    hist_start = last_date - pd.Timedelta(hours=168 + 24)
    history = df[df['datetime'] >= hist_start].copy()
    
    # Future dates
    future_start = last_date + pd.Timedelta(hours=1)
    future_end = last_date + pd.Timedelta(days=days)
    future_dates = pd.date_range(start=future_start, end=future_end, freq='H')
    
    weather_df = generate_future_weather(future_dates)
    predictions = []
    
    print(f"Forecasting {len(future_dates)} hours into the future for {history['zone_id'].nunique()} zones...")
    features_cols = model.feature_name_
    
    for i, current_dt in enumerate(future_dates):
        if i % 24 == 0:
            print(f"[{current_dt}] Day {i//24 + 1}/{days}...")
            
        weather_row = weather_df.iloc[i]
        
        # Build current step dummy records for all zones
        zones = history[['zone_id', 'zone_name', 'zone_type']].drop_duplicates()
        step_df = pd.DataFrame({'zone_id': zones['zone_id']})
        step_df['datetime'] = current_dt
        step_df = step_df.merge(zones, on='zone_id')
        step_df['temperature'] = weather_row['temperature']
        step_df['precipitation'] = weather_row['precipitation']
        step_df['wind_speed'] = weather_row['wind_speed']
        
        from src.config import HOLIDAYS
        step_df['is_holiday'] = np.isin(current_dt.normalize(), HOLIDAYS).astype(int)
        step_df['demand'] = 0 # dummy
        
        # Append to history, compute features, predict
        tmp = pd.concat([history, step_df], ignore_index=True)
        tmp_feat = engineer_features(tmp)
        
        curr_feat = tmp_feat[tmp_feat['datetime'] == current_dt].copy()
        curr_feat = curr_feat.sort_values('zone_id')
        
        X = curr_feat[features_cols]
        preds = model.predict(X)
        preds = np.clip(preds, 0, None)
        
        step_df = step_df.sort_values('zone_id')
        step_df['demand'] = preds
        history = pd.concat([history, step_df], ignore_index=True)
        
        history = history[history['datetime'] >= (current_dt - pd.Timedelta(hours=168+24))]
        
        pred_records = step_df[['datetime', 'zone_id', 'zone_name', 'zone_type', 'demand']].copy()
        pred_records.rename(columns={'demand': 'predicted'}, inplace=True)
        predictions.append(pred_records)
        
    final_df = pd.concat(predictions, ignore_index=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(EVAL_DIR / "future_forecast.csv", index=False)
    print(f"Forecast saved to {EVAL_DIR / 'future_forecast.csv'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30)
    args = parser.parse_args()
    forecast_future(args.days)
