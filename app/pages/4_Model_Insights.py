import streamlit as st
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_loader import load_scored_data

with st.sidebar:
    if st.button("Reload App"):
        st.cache_data.clear()
        st.rerun()

st.set_page_config(layout="wide")
st.title("Model Insights & Agreement")

df = load_scored_data()

required = ["is_if_anomaly", "is_lof_anomaly", "is_pca_anomaly", "risk_score", "final_anomaly"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

st.subheader("Individual Model Behavior")
col1, col2, col3 = st.columns(3)
col1.metric("Isolation Forest Rate", f"{df['is_if_anomaly'].mean() * 100:.2f}%")
col2.metric("LOF Rate", f"{df['is_lof_anomaly'].mean() * 100:.2f}%")
col3.metric("PCA Rate", f"{df['is_pca_anomaly'].mean() * 100:.2f}%")
st.caption("Each model is calibrated to ~1% anomalies for fair comparison. Agreement between models indicates higher confidence.")

st.divider()

df["model_votes"] = df["is_if_anomaly"] + df["is_lof_anomaly"] + df["is_pca_anomaly"]
agreement_stats = {
    "All 3 models agree": (df["model_votes"] == 3).mean(),
    "At least 2 models agree": (df["model_votes"] >= 2).mean(),
    "Only 1 model flags": (df["model_votes"] == 1).mean(),
}

st.subheader("Model Agreement Rates")
st.bar_chart(agreement_stats)
st.caption("Higher agreement = higher confidence of suspicious behavior.")

st.divider()

st.subheader("Risk Score Distribution")
st.area_chart(df["risk_score"].sort_values().reset_index(drop=True))
st.caption("Right tail represents high-risk transactions.")

st.divider()

st.subheader("Confidence Levels")

def confidence_bucket(row):
    if row["model_votes"] == 3 and row["risk_score"] >= df["risk_score"].quantile(0.995):
        return "High Confidence"
    if row["model_votes"] >= 2:
        return "Medium Confidence"
    return "Low Confidence"

df["confidence"] = df.apply(confidence_bucket, axis=1)
confidence_dist = df["confidence"].value_counts(normalize=True)
st.bar_chart(confidence_dist)
st.caption("High confidence alerts are best candidates for manual review.")

st.divider()
st.info("Multiple models agree on a small, high-risk subset — this indicates strong anomaly detection reliability.")
