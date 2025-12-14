import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px

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

    st.subheader("📊 Dataset Summary")
    st.write("Shape of dataset:", df.shape)
    st.write("Missing values per column:")
    st.write(df.isnull().sum())

    st.subheader("🧬 Data Schema (Column Types)")
    st.write(df.dtypes)

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    st.subheader("🔢 Numeric Features Used for Analysis")
    st.write(list(numeric_df.columns))

    if numeric_df.shape[1] == 0:
        st.error("❌ No numeric columns found. Upload a valid dataset.")
        st.stop()

    numeric_df = numeric_df.dropna()

    st.success("✅ Phase 2 completed: Data validated.")

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

    st.success("✅ Phase 3 completed: Features engineered.")

    # =====================================================
    # PHASE 4 — FEATURE SCALING
    # =====================================================

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(engineered_df)

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=engineered_df.columns,
        index=engineered_df.index
    )

    st.success("✅ Phase 4 completed: Features scaled.")

    # =====================================================
    # PHASE 5 — ISOLATION FOREST
    # =====================================================

    st.markdown("---")
    st.header("🌲 Phase 5: Isolation Forest Anomaly Detection")

    contamination = st.slider(
        "Expected Fraud Ratio (Contamination)",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01
    )

    iso_model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42
    )

    iso_model.fit(X_scaled_df)

    # Predictions
    df["iso_anomaly"] = iso_model.predict(X_scaled_df)
    df["iso_score"] = iso_model.decision_function(X_scaled_df)

    # Convert labels
    df["iso_anomaly_label"] = df["iso_anomaly"].map({
        1: "Normal",
        -1: "Anomaly"
    })

    # -------------------------------
    # Results Summary
    # -------------------------------
    st.subheader("📌 Isolation Forest Results Summary")

    anomaly_count = (df["iso_anomaly"] == -1).sum()
    st.write(f"🚨 Detected Anomalies: **{anomaly_count}**")

    # -------------------------------
    # Anomaly Table
    # -------------------------------
    st.subheader("🚨 Flagged Anomalous Transactions")
    st.dataframe(df[df["iso_anomaly"] == -1])

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("📈 Anomaly Score Distribution")

    fig = px.scatter(
        df,
        x=df.index,
        y="iso_score",
        color="iso_anomaly_label",
        title="Isolation Forest Anomaly Scores",
        labels={"iso_score": "Anomaly Score", "index": "Transaction Index"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "✅ Phase 5 completed: Isolation Forest successfully detected anomalies."
    )
