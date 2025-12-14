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
