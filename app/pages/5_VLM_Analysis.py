import streamlit as st
import sys, os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_loader import load_scored_data, load_vlm_results

st.set_page_config(page_title="VLM Analysis", layout="wide")

with st.sidebar:
    if st.button("Reload App"):
        st.cache_data.clear()
        st.rerun()

st.title("Vision-Language Model Analysis")
st.markdown(
    "AI-generated explanations of cardholder transaction behavior "
    "using Vision-Language Model reasoning."
)

df = load_scored_data()
vlm_data = load_vlm_results()

# ========== VLM OVERVIEW ==========
st.divider()
col1, col2, col3 = st.columns(3)

n_analyzed = len(vlm_data)
if n_analyzed > 0:
    avg_suspicion = sum(v["suspicion_score"] for v in vlm_data.values()) / n_analyzed
    high_suspicion = sum(1 for v in vlm_data.values() if v["suspicion_score"] > 0.6)
else:
    avg_suspicion = 0.0
    high_suspicion = 0

col1.metric("Cardholders Analyzed", f"{n_analyzed:,}")
col2.metric("Avg Suspicion Score", f"{avg_suspicion:.3f}")
col3.metric("High Suspicion (>0.6)", f"{high_suspicion:,}")

# ========== VLM RESULTS TABLE ==========
st.divider()
st.subheader("VLM Suspicion Scores by Cardholder")

if vlm_data:
    vlm_df = pd.DataFrame([
        {
            "Cardholder ID": uid,
            "Suspicion Score": v["suspicion_score"],
            "Model": v.get("model", "Unknown"),
            "Explanation Preview": v["explanation"][:120] + "..." if len(v["explanation"]) > 120 else v["explanation"],
        }
        for uid, v in vlm_data.items()
    ]).sort_values("Suspicion Score", ascending=False)

    st.dataframe(vlm_df, use_container_width=True, height=400)

    # ========== SUSPICION DISTRIBUTION ==========
    st.divider()
    st.subheader("Suspicion Score Distribution")
    scores = [v["suspicion_score"] for v in vlm_data.values()]
    score_series = pd.Series(scores, name="suspicion_score")
    st.bar_chart(score_series.value_counts(bins=20).sort_index())

    # ========== DETAILED EXPLANATIONS ==========
    st.divider()
    st.subheader("Detailed AI Explanations")

    selected = st.selectbox(
        "Select a cardholder to view full explanation:",
        options=list(vlm_data.keys()),
        format_func=lambda x: f"Cardholder {x} (Score: {vlm_data[x]['suspicion_score']:.3f})",
    )

    if selected:
        result = vlm_data[selected]
        col_a, col_b = st.columns([1, 3])

        with col_a:
            st.metric("Suspicion Score", f"{result['suspicion_score']:.3f}")
            st.metric("VLM Model", result.get("model", "Unknown"))

            score = result["suspicion_score"]
            if score > 0.6:
                st.error("HIGH RISK")
            elif score > 0.3:
                st.warning("MODERATE RISK")
            else:
                st.success("LOW RISK")

        with col_b:
            st.markdown("**AI-Generated Explanation:**")
            st.markdown(result["explanation"])

        # Show behavior visualization if available
        viz_path = f"data/visualizations/cardholder_{selected}_behavior.png"
        if os.path.exists(viz_path):
            st.divider()
            st.subheader("Behavior Visualization")
            st.image(viz_path, caption=f"Transaction Pattern — Cardholder {selected}")

    # ========== MULTIMODAL IMPACT ==========
    st.divider()
    st.subheader("Multimodal Score Impact")

    if "multimodal_score" in df.columns and "risk_score" in df.columns:
        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("**ML-Only Risk Score Distribution**")
            st.area_chart(df["risk_score"].sort_values().reset_index(drop=True))
        with col_y:
            st.markdown("**Multimodal Score Distribution**")
            st.area_chart(df["multimodal_score"].sort_values().reset_index(drop=True))
else:
    st.info("No VLM analysis results available. Run `python pipeline.py` to generate.")
