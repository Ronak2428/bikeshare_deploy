# Toronto Bikeshare - Deployable API + UI

This repo contains a minimal FastAPI backend and Streamlit frontend to serve and visualize a 30 day Toronto Bikeshare demand forecast.

## What it does
- `/forecast`: Returns a 30 day forecast.
  - If `MODEL_PATH` (pickled SARIMAX) is present, it attempts dynamic forecasting.
  - Else it falls back to reading `forecast_next_30_days.csv`.
- **Streamlit UI**: Plots the forecast and shows a simple table.

## Structure
```
.
├─ app/
│  └─ main.py          # FastAPI
├─ streamlit_app.py     # Streamlit UI
├─ requirements.txt
├─ Procfile             # For Render/Railway/Heroku-style PaaS
├─ runtime.txt          # Optional pinned Python version
└─ Dockerfile           # Container deploy
```

## Local run (two terminals)

Backend
```
pip install -r requirements.txt
export DATA_DIR=/mnt/data
export FORECAST_CSV=$DATA_DIR/forecast_next_30_days.csv
export MODEL_PATH=$DATA_DIR/model_sarimax.pkl
uvicorn app.main:app --reload --port 8000
```

Frontend
```
export API_URL=http://localhost:8000
export FORECAST_CSV=/mnt/data/forecast_next_30_days.csv
streamlit run streamlit_app.py
```

## Render (API)
1) Push these files to GitHub.
2) Create a Web Service on Render from the repo.
3) Build Command: `pip install -r requirements.txt`
4) Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5) Environment Variables:
   - `DATA_DIR=/opt/render/project/src/data` (or leave default)
   - Optionally mount a persistent disk and upload `forecast_next_30_days.csv` and `model_sarimax.pkl`.
6) After deploy, verify: `https://<your-service>.onrender.com/health`

## Hugging Face Spaces (Streamlit)
1) Create a new Space -> Streamlit.
2) Upload this repo.
3) In Secrets, set `API_URL` to your deployed API base URL.
4) Spaces will auto-run `streamlit_app.py`.

## Docker
```
docker build -t bikeshare .
docker run -p 8000:8000 -e DATA_DIR=/data -v $(pwd):/data bikeshare
# API at http://localhost:8000
# To run Streamlit instead:
# docker run -p 8501:8501 -e API_URL=http://host.docker.internal:8000 bikeshare streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## Notes
- If your pickled model was saved differently (for example, not a SARIMAXResults object), modify `app/main.py` to reconstruct before forecasting.
- For CORS, set allowed origins to your frontend domain(s).
- Add auth or keys if you expose this publicly.