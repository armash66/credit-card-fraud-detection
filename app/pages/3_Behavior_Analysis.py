import streamlit as st
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_loader import load_scored_data

with st.sidebar:
    if st.button("Reload App"):
        st.cache_data.clear()
        st.rerun()

st.set_page_config(layout="wide")
st.title("Behavioral Analysis")

df = load_scored_data()

required_cols = ["final_anomaly", "hour", "error_present", "amount", "risk_score"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

anomalies = df[df["final_anomaly"] == 1]
if anomalies.empty:
    st.warning("No anomalies found to analyze.")
    st.stop()

st.subheader("Anomalies by Hour")
hourly = anomalies["hour"].value_counts().sort_index()
if not hourly.empty:
    st.bar_chart(hourly)
else:
    st.info("No hourly pattern detected.")

st.divider()

@st.cache_data(show_spinner=False)
def compute_error_stats(data):
    return data.groupby("error_present")["final_anomaly"].mean().rename({0: "No Error", 1: "Error"})

st.subheader("Error Impact on Fraud Risk")
error_counts = df.groupby("error_present").agg(
    total_transactions=("final_anomaly", "count"),
    anomaly_rate=("final_anomaly", "mean")
).reset_index()
error_counts["error_present"] = error_counts["error_present"].map({0: "No Error", 1: "Error"})
st.dataframe(error_counts, use_container_width=True)
st.caption(
    "Transactions with errors show a much higher anomaly rate, "
    "indicating strong correlation with suspicious behavior."
)

st.divider()

st.subheader("Amount vs Risk (Top Alerts)")
top_risk = (
    df.sort_values("risk_score", ascending=False)
    .dropna(subset=["amount", "risk_score"])
    .head(200)
)
if not top_risk.empty:
    st.scatter_chart(top_risk, x="amount", y="risk_score")
else:
    st.info("Not enough data for risk scatter.")
