import os
import sys
import json
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.inference import CivicMindInference

app = FastAPI(title="CivicMind Dashboard")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

engine = CivicMindInference(k_matches=20)

class SituationInput(BaseModel):
    timestamp: str
    ward: str = "Residential"
    weather_forecast: str = "clear"
    active_conditions: list[str] = []
    event_name: str | None = None
    expected_crowd: int | None = None

@app.get("/", response_class=HTMLResponse)
async def index():
    path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(path) as f:
        return HTMLResponse(f.read())

@app.post("/predict")
async def predict(situation: SituationInput):
    data = situation.model_dump(exclude_none=True)
    result = engine.predict(data)
    body = {
        "query_text": result["query_text"],
        "risk_scores": result["risk_scores"],
        "actions": result["actions"],
        "trace": result["trace"],
        "top_matches": result["top_matches"],
    }
    return JSONResponse(content=json.loads(json.dumps(body, cls=NumpyEncoder)))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
