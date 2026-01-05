import streamlit as st
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from src.data_loader import load_scored_data

with st.sidebar:
    if st.button("🔄 Reload App"):
        st.cache_data.clear()
        st.rerun()

st.set_page_config(layout="wide")
st.title("🔍 Anomaly Explorer")

# ======================================================
# LOAD DATA
# ======================================================
df = load_scored_data()

# ======================================================
# SAFETY CHECKS
# ======================================================
required_cols = [
    "transaction_id",
    "risk_score",
    "amount",
    "hour",
    "error_present",
    "is_if_anomaly",
    "is_lof_anomaly",
    "is_pca_anomaly",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ======================================================
# SIDEBAR CONTROLS
# ======================================================
st.sidebar.header("🔧 Filters")

risk_q = st.sidebar.slider(
    "Risk Threshold (Top %)",
    min_value=0.90,
    max_value=0.995,
    value=0.99,
    step=0.005
)

agreement = st.sidebar.selectbox(
    "Model Agreement",
    ["Any model", "At least 2 models", "All 3 models"]
)

max_rows = st.sidebar.slider(
    "Rows to display",
    min_value=50,
    max_value=300,
    value=100,
    step=50
)

# ======================================================
# FILTER DATA
# ======================================================
risk_threshold = df["risk_score"].quantile(risk_q)
df_f = df[df["risk_score"] >= risk_threshold].copy()

if agreement == "All 3 models":
    df_f = df_f[
        (df_f["is_if_anomaly"] == 1) &
        (df_f["is_lof_anomaly"] == 1) &
        (df_f["is_pca_anomaly"] == 1)
    ]

elif agreement == "At least 2 models":
    df_f["model_votes"] = (
        df_f["is_if_anomaly"] +
        df_f["is_lof_anomaly"] +
        df_f["is_pca_anomaly"]
    )
    df_f = df_f[df_f["model_votes"] >= 2]

df_f = df_f.sort_values("risk_score", ascending=False)

# HARD LIMIT (prevents freezing)
df_f = df_f.head(max_rows).copy()

# ======================================================
# EXPLANATION ENGINE (CACHED)
# ======================================================
def explain_row(row):
    reasons = []

    if row["amount"] > df["amount"].quantile(0.99):
        reasons.append("🔺 Very high transaction amount")

    if row["hour"] <= 5 or row["hour"] >= 23:
        reasons.append("🌙 Unusual transaction time")

    if row["error_present"] == 1:
        reasons.append("⚠️ Transaction error detected")

    model_votes = (
        row["is_if_anomaly"] +
        row["is_lof_anomaly"] +
        row["is_pca_anomaly"]
    )

    if model_votes == 3:
        reasons.append("🤖 Flagged by ALL models")
    elif model_votes == 2:
        reasons.append("🤖 Flagged by multiple models")
    elif model_votes == 1:
        reasons.append("🤖 Flagged by a single model")

    if row["risk_score"] >= df["risk_score"].quantile(0.995):
        reasons.append("🔥 Extremely high overall risk score")

    return reasons


@st.cache_data(show_spinner=False)
def generate_explanations(df_chunk):
    df_chunk = df_chunk.copy()
    df_chunk["why_flagged"] = df_chunk.apply(explain_row, axis=1)
    df_chunk["summary"] = df_chunk["why_flagged"].apply(
        lambda x: " | ".join(x)
    )
    return df_chunk


df_f = generate_explanations(df_f)

# ======================================================
# RISK LEVEL (UI FRIENDLY)
# ======================================================
def risk_label(score):
    if score >= df["risk_score"].quantile(0.995):
        return "🔴 High"
    elif score >= df["risk_score"].quantile(0.99):
        return "🟠 Medium"
    else:
        return "🟢 Low"

df_f["risk_level"] = df_f["risk_score"].apply(risk_label)

# ======================================================
# DISPLAY TABLE
# ======================================================
st.subheader("🚨 High-Risk Transactions")
st.caption("Ranked by risk score. Summaries explain *why* alerts were raised.")
st.info("👇 Scroll down to expand a transaction for full explanation.")

display_cols = [
    "transaction_id",
    "amount",
    "hour",
    "risk_level",
    "risk_score",
    "summary",
]

st.dataframe(
    df_f[display_cols],
    use_container_width=True
)

# ======================================================
# EXPORT
# ======================================================
st.download_button(
    "⬇️ Download Alerts as CSV",
    df_f[display_cols].to_csv(index=False),
    file_name="high_risk_transactions.csv",
    mime="text/csv"
)

# ======================================================
# EXPANDABLE DETAILS
# ======================================================
st.divider()
st.subheader("🧠 Detailed Explanations")

for _, row in df_f.iterrows():
    with st.expander(f"Transaction {int(row['transaction_id'])}"):
        for reason in row["why_flagged"]:
            st.markdown(f"- {reason}")

# ======================================================
# FOOTER METRICS
# ======================================================
st.divider()

col1, col2 = st.columns(2)

col1.metric("Displayed Alerts", f"{len(df_f):,}")
col2.metric(
    "Alert Rate",
    f"{(len(df_f) / len(df)) * 100:.2f}%"
)

st.success("Explainable anomaly detection active 🧠✨")
