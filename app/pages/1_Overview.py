import streamlit as st
import sys
import os

# Allow src imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from src.data_loader import load_scored_data

with st.sidebar:
    if st.button("🔄 Reload App"):
        st.cache_data.clear()
        st.rerun()

st.set_page_config(layout="wide")
st.title("📊 System Overview")

# =========================
# LOAD DATA (CACHED, SAFE)
# =========================
df = load_scored_data()

# =========================
# SAFE METRIC HELPERS
# =========================
def safe_mean(col):
    return df[col].mean() * 100 if col in df.columns else 0.0

def safe_avg(col):
    return df[col].mean() if col in df.columns else 0.0

# =========================
# METRICS
# =========================
total_txns = len(df)
final_rate = safe_mean("final_anomaly")
risk_rate = safe_mean("risk_based_anomaly")
avg_amount = safe_avg("amount")
error_rate = safe_mean("error_present")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total_txns:,}")
col2.metric("Final Anomaly Rate", f"{final_rate:.2f}%")
col3.metric("Avg Transaction Amount", f"${avg_amount:.2f}")
col4.metric("Error Rate", f"{error_rate:.2f}%")

st.divider()

# =========================
# AMOUNT DISTRIBUTION (SAFE)
# =========================
st.subheader("💰 Transaction Amount Distribution")

amount_sample = (
    df["amount"]
    .sample(min(10_000, len(df)), random_state=42)
    .clip(upper=500)
)

st.bar_chart(amount_sample, use_container_width=True)

st.divider()

# =========================
# HOURLY VOLUME
# =========================
if "hour" in df.columns:
    st.subheader("⏰ Transactions by Hour")

    hour_counts = df["hour"].value_counts().sort_index()

    if not hour_counts.empty:
        st.bar_chart(hour_counts)
    else:
        st.info("No hourly data available.")
else:
    st.info("Hour information not available.")

st.divider()

# =========================
# ERROR PRESENCE
# =========================
st.subheader("⚠️ Error Presence")

if "error_present" in df.columns:
    error_counts = (
        df["error_present"]
        .value_counts()
        .rename({0: "No Error", 1: "Error"})
    )

    st.bar_chart(error_counts)
else:
    st.info("Error column not available.")
