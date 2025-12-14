import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="AI Fraud Detection System", layout="wide")

st.title("🚨 AI Fraud & Anomaly Detection System")
st.write("Enterprise-grade fraud detection using ML + Deep Learning")

st.info(
    "Upload a transaction CSV file. "
    "The system uses multiple anomaly detection models and ensemble risk scoring."
)

# -------------------------------------------------
# File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file is not None:

    # -------------------------------------------------
    # Phase 2 — Data Handling
    # -------------------------------------------------
    df = pd.read_csv(uploaded_file)
    numeric_df = df.select_dtypes(include=["int64", "float64"]).dropna()

    if numeric_df.shape[1] == 0:
        st.error("No numeric columns found.")
        st.stop()

    # -------------------------------------------------
    # Phase 3 — Feature Engineering
    # -------------------------------------------------
    engineered_df = numeric_df.copy()

    for col in engineered_df.columns:
        engineered_df[f"{col}_log"] = np.log1p(engineered_df[col])

    z_scores = np.abs(
        (engineered_df - engineered_df.mean()) / engineered_df.std()
    )
    engineered_df["max_z_score"] = z_scores.max(axis=1)
    engineered_df["row_variance"] = engineered_df.var(axis=1)
    engineered_df["transaction_intensity"] = engineered_df.sum(axis=1)

    # -------------------------------------------------
    # Phase 4 — Scaling
    # -------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(engineered_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=engineered_df.columns)

    # -------------------------------------------------
    # Phase 5 — Isolation Forest
    # -------------------------------------------------
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    iso.fit(X_scaled_df)

    df["iso_score"] = -iso.decision_function(X_scaled_df)
    df["iso_flag"] = iso.predict(X_scaled_df) == -1

    # -------------------------------------------------
    # Phase 6 — Autoencoder
    # -------------------------------------------------
    input_dim = X_scaled_df.shape[1]

    autoencoder = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(32, activation="relu"),
        Dense(64, activation="relu"),
        Dense(input_dim, activation="linear")
    ])

    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.fit(
        X_scaled_df, X_scaled_df,
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )

    recon = autoencoder.predict(X_scaled_df, verbose=0)
    df["ae_error"] = np.mean(np.square(X_scaled_df - recon), axis=1)

    # -------------------------------------------------
    # PHASE 7 — ENSEMBLE RISK SCORING
    # -------------------------------------------------
    st.markdown("---")
    st.header("🧠 Phase 7: Ensemble Fraud Risk Scoring")

    # Normalize scores
    df["iso_norm"] = df["iso_score"] / df["iso_score"].max()
    df["ae_norm"] = df["ae_error"] / df["ae_error"].max()

    # Ensemble score (weighted)
    df["fraud_risk_score"] = (
        0.5 * df["iso_norm"] +
        0.5 * df["ae_norm"]
    ) * 100

    # Risk categories
    def risk_label(score):
        if score < 30:
            return "Low Risk"
        elif score < 60:
            return "Medium Risk"
        else:
            return "High Risk"

    df["risk_level"] = df["fraud_risk_score"].apply(_]()
