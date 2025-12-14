# 1️⃣ IMPORTS — TOP OF FILE
from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from model_utils import feature_engineering


# 2️⃣ INITIALIZE FASTAPI — IMMEDIATELY AFTER IMPORTS
app = FastAPI(title="Real-Time Fraud Detection API")


# 3️⃣ MODEL TRAIN / LOAD FUNCTION — BELOW app initialization
def prepare_models():
    ...
    return scaler, iso_model, autoencoder, feature_columns


# 4️⃣ CALL THE FUNCTION (RUNS ON API START)
scaler, iso_model, ae_model, feature_columns = prepare_models()


# 5️⃣ API ENDPOINTS — AT THE VERY BOTTOM
@app.post("/predict")
def predict(transaction: dict):
    ...
    return response
@app.post("/predict")
def predict(transaction: dict):

    # Convert input JSON to DataFrame
    df = pd.DataFrame([transaction])

    # Apply feature engineering
    engineered = feature_engineering(df)

    # Ensure same feature order as training
    engineered = engineered[feature_columns]

    # Scale features
    X_scaled = scaler.transform(engineered)

    # Isolation Forest score
    iso_score = -iso_model.decision_function(X_scaled)[0]

    # Autoencoder reconstruction error
    recon = ae_model.predict(X_scaled, verbose=0)
    ae_error = np.mean((X_scaled - recon) ** 2)

    # Ensemble fraud risk score
    fraud_risk_score = (0.5 * iso_score + 0.5 * ae_error) * 100

    # Risk decision
    if fraud_risk_score > 60:
        decision = "High Risk"
    elif fraud_risk_score > 30:
        decision = "Medium Risk"
    else:
        decision = "Low Risk"

    return {
        "fraud_risk_score": round(fraud_risk_score, 2),
        "decision": decision
    }
