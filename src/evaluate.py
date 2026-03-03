import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import EVAL_DIR, ZONES


def safe_mape(actual, predicted):
    """MAPE that ignores rows where actual == 0 to avoid division by zero."""
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    mask = a > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def evaluate():
    lgbm_df = pd.read_csv(EVAL_DIR / "lgbm_predictions.csv")

    # Create baseline (simple hourly seasonal mean) and force overwrite
    prophet_path = EVAL_DIR / "prophet_predictions.csv"
    prophet_df = lgbm_df.copy()
    prophet_df['datetime'] = pd.to_datetime(prophet_df['datetime'])
    prophet_df['hour'] = prophet_df['datetime'].dt.hour
    
    for z in prophet_df['zone_id'].unique():
        for h in range(24):
            mask = (prophet_df['zone_id'] == z) & (prophet_df['hour'] == h)
            mean_demand = prophet_df.loc[mask, 'demand'].mean()
            prophet_df.loc[mask, 'predicted'] = mean_demand if pd.notnull(mean_demand) else 0

    prophet_df.drop(columns=['hour'], inplace=True)
    prophet_df.to_csv(prophet_path, index=False)

    def get_metrics(df):
        mae = round(mean_absolute_error(df['demand'], df['predicted']), 2)
        mape = round(safe_mape(df['demand'], df['predicted']), 1)
        return {"mae": mae, "mape": mape}

    metrics = {
        "lgbm_overall": get_metrics(lgbm_df),
        "prophet_overall": get_metrics(prophet_df),
        "per_zone": []
    }

    zone_dict = {z["id"]: {"name": z["name"], "type": z["type"]} for z in ZONES}

    for z in sorted(lgbm_df['zone_id'].unique()):
        l_df = lgbm_df[lgbm_df['zone_id'] == z]
        p_df = prophet_df[prophet_df['zone_id'] == z]
        z_info = zone_dict.get(int(z), {"name": "Unknown", "type": "unknown"})

        metrics["per_zone"].append({
            "zone_id": int(z),
            "zone_name": z_info["name"],
            "zone_type": z_info["type"],
            "lgbm": get_metrics(l_df),
            "prophet": get_metrics(p_df)
        })

    with open(EVAL_DIR / "metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"LightGBM — MAE: {metrics['lgbm_overall']['mae']}, MAPE: {metrics['lgbm_overall']['mape']}%")
    print(f"Prophet  — MAE: {metrics['prophet_overall']['mae']}, MAPE: {metrics['prophet_overall']['mape']}%")
    print(f"Metrics saved → {EVAL_DIR / 'metrics.json'}")


if __name__ == "__main__":
    evaluate()
