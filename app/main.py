from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

app = FastAPI(title="Toronto Bikeshare Forecast API", version="1.1.0")

# CORS (loose by default; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/mnt/data"))
FORECAST_CSV = Path(os.environ.get("FORECAST_CSV", DATA_DIR / "forecast_next_30_days.csv"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DATA_DIR / "model_sarimax.pkl"))
METRICS_CSV = Path(os.environ.get("METRICS_CSV", DATA_DIR / "metrics_validation.csv"))

# Serve artifacts (images, etc.) statically
if DATA_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")

class ForecastResponse(BaseModel):
    date: str
    rides: float

@app.get("/health")
def health():
    exists_csv = FORECAST_CSV.exists()
    exists_model = MODEL_PATH.exists()
    exists_metrics = METRICS_CSV.exists()
    return {
        "status": "ok",
        "forecast_csv": bool(exists_csv),
        "model_file": bool(exists_model),
        "metrics_csv": bool(exists_metrics),
        "data_dir": str(DATA_DIR),
    }

def _try_model_forecast(steps: int = 30):
    """Attempt to forecast with a pickled SARIMAXResults-like object."""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        pred = model.get_forecast(steps=steps)
        mean = np.asarray(pred.predicted_mean).tolist()
        try:
            last_date = None
            try:
                last_date = pd.to_datetime(model.data.dates[-1])
            except Exception:
                pass
            if last_date is None:
                last_date = pd.Timestamp.today().normalize()
            dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq="D")
        except Exception:
            dates = pd.date_range(pd.Timestamp.today().normalize(), periods=steps, freq="D")
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "rides": mean})
        return df
    except Exception:
        return None

@app.get("/forecast", response_model=list[ForecastResponse])
def get_forecast():
    # Prefer dynamic model forecast if the model loads; else fall back to the CSV provided
    df = _try_model_forecast(steps=30)
    if df is None:
        if not FORECAST_CSV.exists():
            raise HTTPException(status_code=404, detail="No forecast available (model and CSV missing).")
        try:
            df = pd.read_csv(FORECAST_CSV)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read forecast CSV: {e}")
        cols = [c.lower().strip() for c in df.columns]
        df.columns = cols
        if {"date","rides"}.issubset(set(df.columns)):
            out = df[["date","rides"]].copy()
        elif {"ds","yhat"}.issubset(set(df.columns)):
            out = df.rename(columns={"ds":"date", "yhat":"rides"})[["date","rides"]].copy()
        else:
            date_col = next((c for c in df.columns if "date" in c or c=="ds"), None)
            ride_col = next((c for c in df.columns if "ride" in c or c in ("yhat","forecast","pred")), None)
            if date_col is None or ride_col is None:
                raise HTTPException(status_code=500, detail="CSV must contain date and rides-like columns.")
            out = df.rename(columns={date_col:"date", ride_col:"rides"})[["date","rides"]].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out["rides"] = pd.to_numeric(out["rides"], errors="coerce")
        out = out.dropna()
        return out.to_dict(orient="records")
    else:
        out = df.copy()
        out["rides"] = pd.to_numeric(out["rides"], errors="coerce")
        out = out.dropna()
        return out.to_dict(orient="records")

@app.get("/metrics")
def get_metrics():
    """Return validation metrics from METRICS_CSV if present."""
    if not METRICS_CSV.exists():
        raise HTTPException(status_code=404, detail="metrics_validation.csv not found")
    try:
        df = pd.read_csv(METRICS_CSV)
        # Keep it simple: return all columns and rows
        return {
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
            "shape": df.shape,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics CSV: {e}")