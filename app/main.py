from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging

# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
app = FastAPI(title="Toronto Bikeshare Forecast API", version="1.2.0")

# CORS (loose by default; tighten origins in production if desired)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logger = logging.getLogger("bikeshare-api")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------------------
# Paths & Environment
# (Defaults match Render layout; your ENV overrides on Render still take priority.)
# --------------------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "/opt/render/project/src/data"))
FORECAST_CSV = Path(os.getenv("FORECAST_CSV", str(DATA_DIR / "forecast_next_30_days.csv")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DATA_DIR / "model_sarimax.pkl")))
METRICS_CSV = Path(os.getenv("METRICS_CSV", str(DATA_DIR / "metrics_validation.csv")))

logger.info(f"DATA_DIR:      {DATA_DIR}")
logger.info(f"FORECAST_CSV:  {FORECAST_CSV}")
logger.info(f"MODEL_PATH:    {MODEL_PATH}")
logger.info(f"METRICS_CSV:   {METRICS_CSV}")

# Serve artifacts (images, etc.) statically at /static/*
if DATA_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")
else:
    logger.warning(f"DATA_DIR {DATA_DIR} does not exist; /static will not be mounted.")

# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------
class ForecastResponse(BaseModel):
    date: str
    rides: float

# --------------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------------
def _normalize_forecast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce/rename to ['date','rides'] and clean types."""
    cols = [c.lower().strip() for c in df.columns]
    df = df.copy()
    df.columns = cols

    if {"date", "rides"}.issubset(set(df.columns)):
        out = df[["date", "rides"]].copy()
    elif {"ds", "yhat"}.issubset(set(df.columns)):
        out = df.rename(columns={"ds": "date", "yhat": "rides"})[["date", "rides"]].copy()
    else:
        # Try best-guess columns
        date_col = next((c for c in df.columns if "date" in c or c == "ds"), None)
        ride_col = next((c for c in df.columns if "ride" in c or c in ("yhat", "forecast", "pred")), None)
        if date_col is None or ride_col is None:
            raise HTTPException(status_code=500, detail="CSV must contain date and rides-like columns.")
        out = df.rename(columns={date_col: "date", ride_col: "rides"})[["date", "rides"]].copy()

    # Sanitize
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["rides"] = pd.to_numeric(out["rides"], errors="coerce")
    out = out.dropna()
    return out


def _try_model_forecast(steps: int = 30) -> pd.DataFrame | None:
    """
    Attempt to forecast with a pickled SARIMAXResults-like object.
    If anything fails, return None (the API will fall back to CSV).
    """
    if not MODEL_PATH.exists():
        logger.info("MODEL_PATH not found. Skipping model forecast and using CSV.")
        return None

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        # Most statsmodels SARIMAXResults support get_forecast
        pred = model.get_forecast(steps=steps)
        mean = np.asarray(pred.predicted_mean).tolist()

        # Try to produce dates following the last observed date
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
    except Exception as e:
        logger.exception(f"Model forecast failed: {e}")
        return None

# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/")
def root():
    """Friendly root to avoid a plain 'Not Found'."""
    return {
        "message": "Toronto Bikeshare API is running",
        "endpoints": ["/health", "/forecast", "/metrics", "/static/<file>"],
        "version": app.version,
    }


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


@app.get("/forecast", response_model=list[ForecastResponse])
def get_forecast():
    """
    Prefer dynamic model forecast if the model loads; else fall back to CSV.
    Never 502s just because the model is missing.
    """
    # Try model first
    df_model = _try_model_forecast(steps=30)
    if df_model is not None:
        out = _normalize_forecast_dataframe(df_model)
        return out.to_dict(orient="records")

    # Fall back to CSV
    if not FORECAST_CSV.exists():
        raise HTTPException(status_code=404, detail="No forecast available (model and CSV missing).")
    try:
        df_csv = pd.read_csv(FORECAST_CSV)
    except Exception as e:
        logger.exception(f"Failed to read forecast CSV at {FORECAST_CSV}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read forecast CSV: {e}")

    out = _normalize_forecast_dataframe(df_csv)
    return out.to_dict(orient="records")


@app.get("/metrics")
def get_metrics():
    """Return validation metrics from METRICS_CSV if present."""
    if not METRICS_CSV.exists():
        raise HTTPException(status_code=404, detail="metrics_validation.csv not found")
    try:
        df = pd.read_csv(METRICS_CSV)
        return {
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
            "shape": list(df.shape),
        }
    except Exception as e:
        logger.exception(f"Failed to read metrics CSV at {METRICS_CSV}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read metrics CSV: {e}")
