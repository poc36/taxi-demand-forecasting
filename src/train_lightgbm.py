import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, MODELS_DIR

def train():
    print("Loading engineered features...")
    df = pd.read_csv(DATA_DIR / "taxi_demand_features.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Sort by time
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Features and target
    drop_cols = ['datetime', 'zone_id', 'zone_name', 'zone_type', 'demand']
    features = [c for c in df.columns if c not in drop_cols]
    
    # Train/test split (last 30 days for test)
    test_start = df['datetime'].max() - pd.Timedelta(days=30)
    train_mask = df['datetime'] < test_start
    test_mask = df['datetime'] >= test_start
    
    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, 'demand']
    X_test, y_test = df.loc[test_mask, features], df.loc[test_mask, 'demand']
    
    print(f"Training on {len(X_train)} rows, validating on {len(X_test)} rows")
    
    # Train
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=64,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50)])
    
    # Predict
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)
    
    # Metrics
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    print(f"\n--- Model Performance ---")
    print(f"LGBM MAE: {mae:.2f}")
    print(f"LGBM MAPE: {mape*100:.2f}%")
    
    joblib.dump(model, MODELS_DIR / "lightgbm_model.pkl")
    print(f"Saved model to {MODELS_DIR / 'lightgbm_model.pkl'}")
    
    # Save predictions for evaluation
    res = df.loc[test_mask, ['datetime', 'zone_id', 'demand']].copy()
    res['predicted'] = preds
    output_dir = DATA_DIR.parent / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(output_dir / "lgbm_predictions.csv", index=False)
    print(f"Saved predictions to {output_dir / 'lgbm_predictions.csv'}")

if __name__ == "__main__":
    train()
