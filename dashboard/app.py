from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import json
import uvicorn
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = BASE_DIR / "outputs" / "evaluation"

app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "dashboard" / "templates"))

sys.path.append(str(BASE_DIR))
from src.config import ZONES

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/zones")
def get_zones():
    return ZONES

@app.get("/api/dates")
def get_dates():
    lgbm_df = pd.read_csv(EVAL_DIR / "lgbm_predictions.csv")
    past_dates = pd.to_datetime(lgbm_df['datetime']).dt.date.astype(str).unique().tolist()
    
    future_path = EVAL_DIR / "future_forecast.csv"
    future_dates = []
    if future_path.exists():
        fut_df = pd.read_csv(future_path)
        future_dates = pd.to_datetime(fut_df['datetime']).dt.date.astype(str).unique().tolist()
        
    future_dates = [d for d in future_dates if d not in past_dates]
    
    return {
        "past_dates": sorted(past_dates),
        "future_dates": sorted(future_dates),
        "boundary": past_dates[-1] if past_dates else None
    }

@app.get("/api/metrics")
def get_metrics():
    with open(EVAL_DIR / "metrics.json", "r", encoding='utf-8') as f:
        return json.load(f)

@app.get("/api/heatmap")
def get_heatmap(date: str, hour: int):
    future_path = EVAL_DIR / "future_forecast.csv"
    is_forecast = False
    
    try:
        df = pd.read_csv(EVAL_DIR / "lgbm_predictions.csv")
        df['date'] = pd.to_datetime(df['datetime']).dt.date.astype(str)
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        mask = (df['date'] == date) & (df['hour'] == hour)
        
        if len(df[mask]) == 0 and future_path.exists():
            df = pd.read_csv(future_path)
            df['date'] = pd.to_datetime(df['datetime']).dt.date.astype(str)
            df['hour'] = pd.to_datetime(df['datetime']).dt.hour
            mask = (df['date'] == date) & (df['hour'] == hour)
            is_forecast = True
            
        subset = df[mask]
        records = []
        zone_dict = {z["id"]: {"name": z["name"], "type": z["type"]} for z in ZONES}
        
        for _, row in subset.iterrows():
            z_id = int(row['zone_id'])
            records.append({
                "zone_id": z_id,
                "zone_name": zone_dict[z_id]["name"],
                "zone_type": zone_dict[z_id]["type"],
                "predicted": float(row['predicted']),
                "actual": float(row['demand']) if not is_forecast and 'demand' in row else None
            })
            
        return {"data": records, "is_forecast": is_forecast}
    except Exception as e:
        return {"data": [], "is_forecast": False, "error": str(e)}

@app.get("/api/timeseries")
def get_timeseries(zone_id: int, include_future: bool = True):
    try:
        lgbm_df = pd.read_csv(EVAL_DIR / "lgbm_predictions.csv")
        prophet_path = EVAL_DIR / "prophet_predictions.csv"
        prophet_df = pd.read_csv(prophet_path) if prophet_path.exists() else pd.DataFrame()
        
        future_path = EVAL_DIR / "future_forecast.csv"
        fut_df = pd.read_csv(future_path) if future_path.exists() and include_future else pd.DataFrame()
        
        def process(df, is_fut=False):
            if len(df) == 0: return df
            sub = df[df['zone_id'] == zone_id].sort_values('datetime')
            sub['datetime'] = pd.to_datetime(sub['datetime']).dt.strftime('%m-%d %H:%M')
            return sub
            
        l_sub = process(lgbm_df)
        p_sub = process(prophet_df)
        f_sub = process(fut_df, True)
        
        res = {
            "past": {
                "labels": l_sub['datetime'].tolist(),
                "actual": l_sub['demand'].tolist() if 'demand' in l_sub else [],
                "lgbm": l_sub['predicted'].tolist() if 'predicted' in l_sub else [],
                "prophet": p_sub['predicted'].tolist() if not p_sub.empty and 'predicted' in p_sub else []
            },
            "future": {
                "labels": f_sub['datetime'].tolist() if not f_sub.empty else [],
                "predicted": f_sub['predicted'].tolist() if not f_sub.empty else []
            },
            "boundary_idx": len(l_sub)
        }
        return res
    except Exception as e:
        return {"past": {"labels": [], "actual": [], "lgbm": [], "prophet": []}, "future": {"labels": [], "predicted": []}, "boundary_idx": 0}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8050, reload=True)
