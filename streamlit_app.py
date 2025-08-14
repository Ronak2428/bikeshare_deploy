import os
import glob
import pandas as pd
import requests
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Toronto Bikeshare Forecast", layout="wide")

st.title("Toronto Bikeshare - 30 Day Forecast")
st.caption("FastAPI + Streamlit demo.")

api_url = os.environ.get("API_URL", "http://localhost:8000")
data_dir = os.environ.get("DATA_DIR", "/mnt/data")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=api_url, help="Point this to your deployed API base URL.")
    st.markdown("Endpoints")
    st.code(f"{api_url}/health\n{api_url}/forecast\n{api_url}/metrics\n{api_url}/static/<file>")
    if st.button("Check API Health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            st.json(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")

tab1, tab2, tab3 = st.tabs(["Forecast", "Metrics", "Artifacts"])

with tab1:
    try:
        r = requests.get(f"{api_url}/forecast", timeout=15)
        r.raise_for_status()
        data = pd.DataFrame(r.json())
        data["date"] = pd.to_datetime(data["date"])
        st.line_chart(data.set_index("date")["rides"])
        st.dataframe(data, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning("Could not fetch from API. Attempting local CSV fallback...")
        local_csv = os.environ.get("FORECAST_CSV", "/mnt/data/forecast_next_30_days.csv")
        try:
            df = pd.read_csv(local_csv)
            cols = [c.lower().strip() for c in df.columns]
            df.columns = cols
            if {"date","rides"}.issubset(set(df.columns)):
                data = df[["date","rides"]].copy()
            elif {"ds","yhat"}.issubset(set(df.columns)):
                data = df.rename(columns={"ds":"date", "yhat":"rides"})[["date","rides"]].copy()
            else:
                raise ValueError("CSV must contain date/rides or ds/yhat")
            data["date"] = pd.to_datetime(data["date"])
            st.line_chart(data.set_index("date")["rides"])
            st.dataframe(data, use_container_width=True, hide_index=True)
        except Exception as ee:
            st.error(f"Failed to load any forecast. {ee}")

with tab2:
    st.subheader("Validation Metrics")
    try:
        r = requests.get(f"{api_url}/metrics", timeout=10)
        r.raise_for_status()
        payload = r.json()
        dfm = pd.DataFrame(payload["rows"])
        st.dataframe(dfm, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning("Could not fetch metrics from API. Attempting local CSV...")
        local_metrics = os.environ.get("METRICS_CSV", "/mnt/data/metrics_validation.csv")
        try:
            dfm = pd.read_csv(local_metrics)
            st.dataframe(dfm, use_container_width=True, hide_index=True)
        except Exception as ee:
            st.info("No metrics available.")

with tab3:
    st.subheader("Artifacts (images)")
    # If API is remote, display via /static URLs; otherwise read locally
    # Try remote first
    try:
        # naive list of common filenames the user uploaded
        candidates = ["WhatsApp Image 2025-08-13 at 12.03.52 AM.jpeg",
                      "WhatsApp Image 2025-08-13 at 12.03.52 AM (1).jpeg"]
        shown = False
        for name in candidates:
            url = f"{api_url}/static/{name}"
            st.image(url, caption=name, use_column_width=True)
            shown = True
        if not shown:
            raise RuntimeError("No remote images listed.")
    except Exception:
        # local glob
        imgs = []
        for ext in ("*.png","*.jpg","*.jpeg"):
            imgs.extend(glob.glob(os.path.join(data_dir, ext)))
        if imgs:
            for p in imgs:
                st.image(p, caption=os.path.basename(p), use_column_width=True)
        else:
            st.info("No images found in DATA_DIR.")