import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Fraud Detection System",
    layout="wide"
)

st.title("🚨 AI Fraud & Anomaly Detection System")
st.write("Upload transaction data to detect suspicious behavior.")

uploaded_file = st.file_uploader(
    "Upload Transaction CSV File",
    type=["csv"]
)
