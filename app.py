import streamlit as st
import pandas as pd
import numpy as np

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

    st.subheader("🔍 Numeric Data Preview")
    st.dataframe(numeric_df.head())

    st.success("✅ Phase 2 completed: Data validated successfully.")

    # =====================================================
    # PHASE 3 — FEATURE ENGINEERING
    # =====================================================

    st.markdown("---")
    st.header("🧠 Phase 3: Feature Engineering")

    st.write(
        "This phase transforms raw numeric values into "
        "behaviour-based features used for fraud detection."
    )

    engineered_df = numeric_df.copy()

    # -------------------------------
    # Feature 1: Log Transformation
    # -------------------------------
    for col in engineered_df.columns:
        engineered_df[f"{col}_log"] = np.log1p(engineered_df[col])

    # -------------------------------
    # Feature 2: Z-Score (Deviation)
    # -------------------------------
    z_scores = np.abs(
        (engineered_df - engineered_df.mean()) / engineered_df.std()
    )
    engineered_df["max_z_score"] = z_scores.max(axis=1)

    # -------------------------------
    # Feature 3: Row-wise Variance (Behaviour Spread)
    # -------------------------------
    engineered_df["row_variance"] = engineered_df.var(axis=1)

    # -------------------------------
    # Feature 4: Transaction Intensity
    # -------------------------------
    engineered_df["transaction_intensity"] = engineered_df.sum(axis=1)

    # -------------------------------
    # Preview Engineered Features
    # -------------------------------
    st.subheader("🧪 Engineered Feature Preview")
    st.dataframe(engineered_df.head())

    st.subheader("📐 Engineered Feature Summary")
    st.write(engineered_df.describe())

    st.success(
        "✅ Phase 3 completed: Behavioural features engineered successfully."
    )
