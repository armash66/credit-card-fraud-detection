import streamlit as st
import sys, os, json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_loader import load_scored_data

st.set_page_config(page_title="Explainability", layout="wide")

with st.sidebar:
    if st.button("Reload App"):
        st.cache_data.clear()
        st.rerun()

st.title("Explainability & Feature Analysis")
st.markdown(
    "Understand *why* transactions are flagged using SHAP values, "
    "feature importance, and per-transaction explanations."
)

df = load_scored_data()

# ========== FEATURE IMPORTANCE ==========
st.divider()
st.subheader("Feature Importance")

fi_path = "data/visualizations/feature_importance.png"
if os.path.exists(fi_path):
    st.image(fi_path, caption="Feature Importance (SHAP / Permutation)")
else:
    st.info("Feature importance plot not yet generated. Run `python pipeline.py`.")

# ========== SHAP SUMMARY ==========
shap_path = "data/visualizations/shap_summary.png"
if os.path.exists(shap_path):
    st.divider()
    st.subheader("SHAP Summary Plot")
    st.image(shap_path, caption="SHAP Feature Impact on Anomaly Score")

# ========== EVALUATION METRICS ==========
st.divider()
st.subheader("Model Evaluation Metrics")

eval_path = "data/evaluation_report.json"
if os.path.exists(eval_path):
    with open(eval_path, "r") as f:
        eval_data = json.load(f)

    eval_df = pd.DataFrame(eval_data)
    display_cols = [c for c in ["model", "precision", "recall", "f1", "roc_auc", "fpr"] if c in eval_df.columns]
    st.dataframe(eval_df[display_cols], use_container_width=True)

    if "model" in eval_df.columns:
        chart_df = eval_df.set_index("model")[["precision", "recall", "f1"]].T
        st.bar_chart(chart_df)

    roc_path = "data/visualizations/roc_curves.png"
    if os.path.exists(roc_path):
        st.divider()
        st.subheader("ROC Curves")
        st.image(roc_path, caption="ROC Curves — Baseline vs Multimodal")

    pr_path = "data/visualizations/pr_curves.png"
    if os.path.exists(pr_path):
        st.subheader("Precision-Recall Curves")
        st.image(pr_path, caption="Precision-Recall — Baseline vs Multimodal")
else:
    st.info("Evaluation report not available. Labels needed for evaluation.")

# ========== PER-TRANSACTION EXPLANATION ==========
st.divider()
st.subheader("Transaction-Level Explanations")

required = ["transaction_id", "amount", "risk_score"]
if all(c in df.columns for c in required):
    top_n = st.slider("Number of top-risk transactions", 10, 100, 30)

    score_col = "multimodal_score" if "multimodal_score" in df.columns else "risk_score"
    top_df = df.nlargest(top_n, score_col)

    display = ["transaction_id", "amount", "hour"]
    if "risk_score" in df.columns: display.append("risk_score")
    if "multimodal_score" in df.columns: display.append("multimodal_score")
    if "fraud_probability" in df.columns: display.append("fraud_probability")
    if "multimodal_risk_level" in df.columns: display.append("multimodal_risk_level")
    if "confidence" in df.columns: display.append("confidence")

    available = [c for c in display if c in top_df.columns]
    st.dataframe(top_df[available], use_container_width=True)

    st.markdown("**Click a transaction for detailed explanation:**")
    for _, row in top_df.head(20).iterrows():
        tid = int(row.get("transaction_id", 0))
        score = row.get(score_col, 0)
        with st.expander(f"Transaction {tid} — Score: {score:.4f}"):
            reasons = []
            if row.get("amount", 0) > df["amount"].quantile(0.99):
                reasons.append("Very high transaction amount")
            h = row.get("hour", 12)
            if h <= 5 or h >= 23:
                reasons.append("Unusual transaction time")
            if row.get("error_present", 0) == 1:
                reasons.append("Transaction error")

            acols = [c for c in row.index if c.endswith("_anomaly") and c not in ("final_anomaly", "risk_based_anomaly")]
            votes = sum(row.get(c, 0) for c in acols)
            if votes >= 3: reasons.append("ALL models flagged")
            elif votes >= 2: reasons.append("Multiple models flagged")
            elif votes >= 1: reasons.append("One model flagged")

            if row.get("multimodal_score", 0) > 0.7:
                reasons.append("Critical multimodal score")

            vlm = row.get("vlm_explanation", "")
            if vlm and vlm != "No VLM analysis available" and vlm != "VLM analysis not available":
                reasons.append(f"AI Analysis: {vlm[:200]}")

            if not reasons:
                reasons.append("Within normal parameters")

            for r in reasons:
                st.markdown(f"- {r}")

            mc, mc2 = st.columns(2)
            mc.metric("ML Risk Score", f"{row.get('risk_score', 0):.4f}")
            mc2.metric("VLM Suspicion", f"{row.get('vlm_suspicion_score', 0):.4f}")
else:
    st.warning("Required columns missing from data.")

overview_path = "data/visualizations/system_overview.png"
if os.path.exists(overview_path):
    st.divider()
    st.subheader("System Overview")
    st.image(overview_path, caption="System-Wide Fraud Detection Overview")

st.divider()
st.info("Explainability module active — Transparent AI for trustworthy fraud detection.")
