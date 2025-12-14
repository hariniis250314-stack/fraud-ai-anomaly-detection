import streamlit as st
import pandas as pd

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

    # STEP 2.2 — Load & Preview Data
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # STEP 2.3 — Basic Data Validation
    st.subheader("📊 Dataset Summary")
    st.write("Shape of dataset:", df.shape)

    st.write("Missing values per column:")
    st.write(df.isnull().sum())

    # STEP 2.8 — Schema Detection
    st.subheader("🧬 Data Schema (Column Types)")
    st.write(df.dtypes)

    # STEP 2.4 — Numeric Feature Detection
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    st.subheader("🔢 Numeric Features Used for Analysis")
    st.write(list(numeric_df.columns))

    # STEP 2.5 — Empty Numeric Guardrail
    if numeric_df.shape[1] == 0:
        st.error(
            "❌ No numeric columns found. "
            "Please upload a valid transaction dataset with numeric values."
        )
        st.stop()

    # STEP 2.9 — Numeric Data Preview
    st.subheader("🔍 Numeric Data Preview")
    st.dataframe(numeric_df.head())

    # STEP 2.6 — Clean Numeric Data
    numeric_df = numeric_df.dropna()

    # STEP 2.10 — Phase Completion Indicator
    st.success(
        "✅ Phase 2 completed: Data uploaded, validated, "
        "and schema detected successfully."
    )
