from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import time

app = FastAPI(title="Fraud Detection API", version="1.0")

model = joblib.load("models/fraud_model.joblib")
scaler = joblib.load("models/scaler.joblib")


class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction):
    start = time.time()

    data = txn.model_dump()
    amount_scaled = scaler.transform([[data.pop("Amount")]])[0][0]
    time_scaled = scaler.transform([[data.pop("Time")]])[0][0]

    features = pd.DataFrame([{
        **data,
        "Amount_scaled": amount_scaled,
        "Time_scaled": time_scaled,
    }])[model.feature_names_in_]

    proba = model.predict_proba(features)[0][1]
    latency = (time.time() - start) * 1000

    return PredictionResponse(
        fraud_probability=round(float(proba), 4),
        is_fraud=bool(proba >= 0.5),
        latency_ms=round(latency, 2)
    )
