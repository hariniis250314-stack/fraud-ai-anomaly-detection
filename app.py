import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Fraud Detection System",
    layout="wide"
)

st.title("🚨 AI Fraud & Anomaly Detection System")
st.write("Detecting suspicious transactions using advanced ML and DL models.")

st.info(
    "📌 Upload a CSV file containing transaction data. "
    "The system will validate the data and automatically detect usable features."
)

# -------------------------------
# CSV Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Transaction CSV File",
    type=["csv"]
)

# -------------------------------
# Phase 2: Data Handling
# -------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    numeric_df = df.select_dtypes(include=["int64", "float64"]).dropna()

    if numeric_df.shape[1] == 0:
        st.error("❌ No numeric columns found.")
        st.stop()

    # =====================================================
    # PHASE 3 — FEATURE ENGINEERING
    # =====================================================

    engineered_df = numeric_df.copy()

    for col in engineered_df.columns:
        engineered_df[f"{col}_log"] = np.log1p(engineered_df[col])

    z_scores = np.abs(
        (engineered_df - engineered_df.mean()) / engineered_df.std()
    )
    engineered_df["max_z_score"] = z_scores.max(axis=1)
    engineered_df["row_variance"] = engineered_df.var(axis=1)
    engineered_df["transaction_intensity"] = engineered_df.sum(axis=1)

    # =====================================================
    # PHASE 4 — FEATURE SCALING
    # =====================================================

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(engineered_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=engineered_df.columns)

    # =====================================================
    # PHASE 5 — ISOLATION FOREST
    # =====================================================

    iso = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42
    )
    iso.fit(X_scaled_df)

    df["iso_label"] = iso.predict(X_scaled_df)
    df["iso_score"] = iso.decision_function(X_scaled_df)

    # =====================================================
    # PHASE 6 — AUTOENCODER
    # =====================================================

    st.markdown("---")
    st.header("🧠 Phase 6: Deep Autoencoder Anomaly Detection")

    st.write(
        "The autoencoder learns normal transaction patterns and "
        "flags records with high reconstruction error."
    )

    input_dim = X_scaled_df.shape[1]

    autoencoder = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(32, activation="relu"),
        Dense(64, activation="relu"),
        Dense(input_dim, activation="linear")
    ])

    autoencoder.compile(
        optimizer="adam",
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True
    )

    autoencoder.fit(
        X_scaled_df,
        X_scaled_df,
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )

    # Reconstruction error
    reconstructions = autoencoder.predict(X_scaled_df, verbose=0)
    reconstruction_error = np.mean(
        np.square(X_scaled_df - reconstructions),
        axis=1
    )

    df["ae_error"] = reconstruction_error

    # Threshold based on percentile
    threshold = np.percentile(reconstruction_error, 95)
    df["ae_anomaly"] = df["ae_error"] > threshold

    # -------------------------------
    # Autoencoder Results
    # -------------------------------
    st.subheader("📌 Autoencoder Results Summary")

    ae_count = df["ae_anomaly"].sum()
    st.write(f"🚨 Detected Anomalies (Autoencoder): **{ae_count}**")

    st.subheader("🚨 Autoencoder Flagged Transactions")
    st.dataframe(df[df["ae_anomaly"] == True])

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("📈 Reconstruction Error Distribution")

    fig = px.scatter(
        df,
        x=df.index,
        y="ae_error",
        color=df["ae_anomaly"].map({True: "Anomaly", False: "Normal"}),
        title="Autoencoder Reconstruction Errors",
        labels={"ae_error": "Reconstruction Error"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "✅ Phase 6 completed: Autoencoder successfully detected anomalies."
    )
