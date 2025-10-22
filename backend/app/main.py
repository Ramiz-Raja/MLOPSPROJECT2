# backend/app/main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from .model_manager import load_model_from_wandb

app = FastAPI(title="Iris inference service")

class PredictRequest(BaseModel):
    # expects list of 4 floats (sepal/petal measurements)
    features: list

@app.on_event("startup")
def startup_load_model():
    global model
    wandb_project = os.getenv("WANDB_PROJECT", "mlops-capstone")
    wandb_entity = os.getenv("WANDB_ENTITY", None)
    try:
        model = load_model_from_wandb(wandb_entity, wandb_project)
        app.state.model_ready = True
    except Exception as e:
        # If model not available, mark not ready but keep service up.
        model = None
        app.state.model_ready = False
        app.state.load_error = str(e)

@app.get("/health")
def health():
    if app.state.model_ready:
        return {"status": "ok", "model": True}
    else:
        return {"status": "degraded", "model": False, "error": getattr(app.state, "load_error", None)}

@app.post("/predict")
def predict(req: PredictRequest):
    if not app.state.model_ready:
        return {"error": "model not ready", "details": getattr(app.state, "load_error", None)}
    features = np.array(req.features).reshape(1, -1)
    pred = model.predict(features).tolist()
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features).tolist()
    return {"prediction": pred, "probability": proba}
