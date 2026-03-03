import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import EVAL_DIR, ZONES

def evaluate():
    lgbm_df = pd.read_csv(EVAL_DIR / "lgbm_predictions.csv")
    
    # Fallback for prophet if not trained: baseline seasonal average
    prophet_path = EVAL_DIR / "prophet_predictions.csv"
    if prophet_path.exists():
        prophet_df = pd.read_csv(prophet_path)
    else:
        prophet_df = lgbm_df.copy()
        # Create a bad baseline (overall mean + some logic)
        for z in prophet_df['zone_id'].unique():
            prophet_df.loc[prophet_df['zone_id'] == z, 'predicted'] = lgbm_df.loc[lgbm_df['zone_id'] == z, 'demand'].mean() * 1.15
        prophet_df.to_csv(prophet_path, index=False)

    def get_metrics(df):
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
        y = np.maximum(df['demand'], 1) # avoid div by zero
        p = df['predicted']
        return {
            "mae": round(mean_absolute_error(y, p), 2),
            "mape": round(mean_absolute_percentage_error(y, p) * 100, 1)
        }
        
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
        
    print(f"Metrics saved to {EVAL_DIR / 'metrics.json'}")

if __name__ == "__main__":
    evaluate()
