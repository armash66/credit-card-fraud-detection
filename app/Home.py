import streamlit as st
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import reset_data_cache

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

with st.sidebar:
    st.markdown("### System Controls")
    if st.button("Reload Dashboard"):
        reset_data_cache()
        st.rerun()

st.title("Multimodal Credit Card Fraud Detection System")

st.markdown("""
### What is this system?

This application detects **suspicious credit card transactions** using a **hybrid multimodal approach** combining:

- **Unsupervised ML Models** — Isolation Forest, Autoencoder, One-Class SVM, LOF
- **Behavior Visualization** — Spending heatmaps, transaction timelines, merchant patterns
- **Vision-Language Model (VLM)** — AI-generated explanations of suspicious behavior
- **Multimodal Score Fusion** — Combines ML anomaly scores + VLM suspicion scores
- **Explainable AI** — SHAP values, feature importance, per-transaction reasoning

Designed for **real-world fraud detection scenarios** where fraud labels are rare or unavailable.
""")

st.divider()

# ========== DETECTION MODELS ==========
st.subheader("Detection Models")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    **Isolation Forest**
    - Detects globally rare patterns
    - Tree-based isolation
    """)
with c2:
    st.markdown("""
    **Autoencoder (PyTorch)**
    - Reconstruction error
    - Deep learning-based
    """)
with c3:
    st.markdown("""
    **One-Class SVM**
    - Decision boundary learning
    - Kernel-based detection
    """)
with c4:
    st.markdown("""
    **Local Outlier Factor**
    - Density-based anomalies
    - Local neighborhood analysis
    """)

st.divider()

# ========== MULTIMODAL PIPELINE ==========
st.subheader("Multimodal Pipeline")
st.markdown("""
```
Raw Data -> Preprocessing -> 4 ML Models -> Ensemble Score
                                              |
Behavior Visualization -> VLM Analysis -> Suspicion Score
                                              |
              Multimodal Fusion: 0.6 x ML + 0.4 x VLM = Final Score
                                              |
              SHAP Explainability -> Dashboard -> Fraud Alerts
```
""")

st.divider()

# ========== RISK SCORING ==========
st.subheader("Risk Scoring Strategy")
st.markdown("""
The system combines **multiple signals** for maximum reliability:

- **4 anomaly detection models** with weighted ensemble scoring
- **VLM-powered** behavioral pattern analysis
- **Multimodal fusion** — `final_score = 0.6 x ML + 0.4 x VLM`
- **SHAP-based** transparency for every flagged transaction

This reduces false positives and provides **actionable, explainable alerts**.
""")

st.divider()

# ========== NAVIGATION ==========
st.subheader("How to Navigate")
st.markdown("""
Use the sidebar to explore:

| Page | Purpose |
|------|---------|
| **Overview** | System-wide statistics and distributions |
| **Anomaly Explorer** | Inspect and filter suspicious transactions |
| **Behavior Analysis** | Temporal and categorical anomaly patterns |
| **Model Insights** | Compare models and agreement analysis |
| **VLM Analysis** | AI-generated explanations with behavior visuals |
| **Explainability** | SHAP values, feature importance, evaluation metrics |
""")

st.info("Multimodal Fraud Detection System Ready — Use the sidebar to begin.")