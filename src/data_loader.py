"""
Centralized Data Loader (Enhanced)
===================================
Safe, cached data access for the Streamlit dashboard.
Supports both baseline and multimodal scored data.
"""
import pandas as pd
import os
import json
import streamlit as st

DATA_PATH = "data/scored_transactions.parquet"
VLM_PATH = "data/vlm_results.json"

@st.cache_data(show_spinner=False)
def load_scored_data():
    if not os.path.exists(DATA_PATH):
        st.error("Scored data not found. Run `python pipeline.py` first.")
        st.stop()
    df = pd.read_parquet(DATA_PATH)
    defaults = {
        "final_anomaly": 0, "risk_score": 0.0, "risk_based_anomaly": 0,
        "is_if_anomaly": 0, "is_lof_anomaly": 0, "is_pca_anomaly": 0,
        "all_3_agree": 0, "any_2_agree": 0, "error_present": 0,
        "hour": 0, "amount": 0.0, "multimodal_score": 0.0,
        "fraud_probability": 0.0, "vlm_suspicion_score": 0.0,
        "vlm_explanation": "", "multimodal_risk_level": "N/A",
        "confidence": "None", "model_votes": 0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df

@st.cache_data(show_spinner=False)
def load_vlm_results():
    if not os.path.exists(VLM_PATH):
        return {}
    with open(VLM_PATH, "r") as f:
        return json.load(f)

def reset_data_cache():
    st.cache_data.clear()
