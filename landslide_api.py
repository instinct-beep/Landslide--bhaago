"""
landslide_api.py
Run:  uvicorn landslide_api:app --reload --port 8000
"""
import asyncio
import json
from contextlib import asynccontextmanager
import joblib
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL = None
STATS = None

FEATURE_COLS = [
    "Aspect_z", "Dist_River_z", "Elevation_z",
    "FlowAccumulation_z", "NDVI_z", "Rainfall_z", "Slope_z",
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, STATS
    try:
        MODEL = joblib.load("landslide_rf_model.pkl")
        with open("feature_stats.json", "r") as f:
            STATS = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load model/stats: {e}")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)

async def _fetch_json(client, url):
    try:
        r = await client.get(url, timeout=8.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def z_score(val, col_name):
    if col_name not in STATS or STATS[col_name]["std"] == 0:
        return 0.0
    return (val - STATS[col_name]["mean"]) / STATS[col_name]["std"]

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
async def predict(req: PredictRequest):
    if MODEL is None or STATS is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    lat, lng = req.lat, req.lng
    
    # Defaults (mean z-score = 0) for features we can't fetch live
    features = {k: 0.0 for k in FEATURE_COLS}
    sources = {k: "estimated" for k in FEATURE_COLS}
    live_raw = {}

    # Fetch Live Data
    elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat},{lat+0.01},{lat}&longitude={lng},{lng},{lng+0.01}"
    rain_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&daily=precipitation_sum&past_days=7&forecast_days=0"

    async with httpx.AsyncClient() as client:
        elev_r, rain_r = await asyncio.gather(
            _fetch_json(client, elev_url),
            _fetch_json(client, rain_url)
        )

    if elev_r and isinstance(elev_r.get("elevation"), list):
        elevs = elev_r["elevation"]
        if len(elevs) == 3 and all(e is not None for e in elevs):
            c, n, e = float(elevs[0]), float(elevs[1]), float(elevs[2])
            dx = (e - c) / 1110.0
            dy = (n - c) / 1110.0
            slope = float(np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy))))
            aspect = float((np.degrees(np.arctan2(-dx, -dy)) + 360.0) % 360.0)
            
            live_raw["elevation"] = c
            live_raw["slope_deg"] = slope
            live_raw["aspect_c"] = aspect

    if rain_r and isinstance(rain_r.get("daily", {}).get("precipitation_sum"), list):
        live_raw["rainfall"] = sum(v or 0 for v in rain_r["daily"]["precipitation_sum"])

    # Convert to Z-scores using training stats
    if "elevation" in live_raw:
        features["Elevation_z"] = z_score(live_raw["elevation"], "elevation")
        features["Slope_z"] = z_score(live_raw["slope_deg"], "slope_deg")
        features["Aspect_z"] = z_score(live_raw["aspect_c"], "aspect_c")
        sources["Elevation_z"] = sources["Slope_z"] = sources["Aspect_z"] = "live"

    if "rainfall" in live_raw:
        # If rainfall isn't in stats (e.g. missing raw col), fallback to generic
        if "rainfall" in STATS:
            features["Rainfall_z"] = z_score(live_raw["rainfall"], "rainfall")
        else:
            features["Rainfall_z"] = (live_raw["rainfall"] - 50) / 100 # rough fallback
        sources["Rainfall_z"] = "live"

    # Predict
    X = np.array([[features[f] for f in FEATURE_COLS]], dtype=np.float64)
    proba = float(MODEL.predict_proba(X)[0, 1])

    label = "CRITICAL" if proba > 0.75 else "HIGH" if proba > 0.5 else "MODERATE" if proba > 0.25 else "LOW"

    return {
        "risk": proba,
        "label": label,
        "features": features,
        "sources": sources,
        "raw_live_values": live_raw, # Send raw numbers so UI shows human-readable units
        "coords": {"lat": lat, "lng": lng}
    }