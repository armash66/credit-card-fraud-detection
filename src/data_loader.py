import pandas as pd
import os
import streamlit as st

DATA_PATH = "data/scored_transactions.parquet"

@st.cache_data(show_spinner=False)
def load_scored_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ Scored data not found. Run main.py first.")
        st.stop()

    df = pd.read_parquet(DATA_PATH)

    # Defensive defaults (prevents crashes)
    defaults = {
        "final_anomaly": 0,
        "risk_score": 0.0,
        "risk_based_anomaly": 0,
        "is_if_anomaly": 0,
        "is_lof_anomaly": 0,
        "is_pca_anomaly": 0,
        "all_3_agree": 0,
        "any_2_agree": 0,
        "error_present": 0,
        "hour": 0,
        "amount": 0.0,
    }

    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    return df
