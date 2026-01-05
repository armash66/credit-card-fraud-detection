import streamlit as st
import sys
import os

# Allow imports from src/
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.data_loader import reset_data_cache

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# =========================
# SIDEBAR CONTROLS (IMPORTANT)
# =========================
with st.sidebar:
    st.markdown("### 🔧 System Controls")

    if st.button("🔄 Reload Dashboard"):
        reset_data_cache()
        st.rerun()

st.title("💳 Credit Card Fraud Detection System")

st.markdown(
    """
    ### 🧠 What is this system?

    This application detects **suspicious credit card transactions**
    using **unsupervised machine learning** techniques.

    It is designed for **real-world fraud detection scenarios** where:
    - Fraud labels are rare or unavailable
    - Patterns evolve over time
    - Interpretability matters as much as accuracy
    """
)

st.divider()

# =========================
# MODELS USED
# =========================
st.subheader("🔍 Detection Models Used")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        **Isolation Forest**
        - Detects rare & unusual patterns
        - Works well on large datasets
        """
    )

with col2:
    st.markdown(
        """
        **Local Outlier Factor (LOF)**
        - Finds local density anomalies
        - Good for subtle fraud patterns
        """
    )

with col3:
    st.markdown(
        """
        **PCA Reconstruction Error**
        - Detects abnormal behavior
        - Based on information loss
        """
    )

st.divider()

# =========================
# RISK SCORING
# =========================
st.subheader("⚖️ Risk Scoring Strategy")

st.markdown(
    """
    Instead of trusting a single model, this system:

    - Combines **multiple anomaly detectors**
    - Computes a **weighted risk score**
    - Flags only **high-confidence suspicious transactions**

    This reduces false positives and improves trust.
    """
)

st.divider()

# =========================
# HOW TO USE
# =========================
st.subheader("🧭 How to Navigate")

st.markdown(
    """
    Use the sidebar to explore:

    - **Overview** → System-wide statistics  
    - **Anomaly Explorer** → Inspect suspicious transactions  
    - **Behavior Analysis** → Understand fraud patterns  
    - **Model Insights** → Compare model behavior & agreement  

    This app is built for **analysts**, not just models.
    """
)

st.success("✅ System Ready — Use the sidebar to begin exploration.")