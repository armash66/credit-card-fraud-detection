"""
Explainability Module
======================
SHAP values, feature importance, and per-transaction explanations.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def compute_shap_values(model, X_scaled, feature_names, max_samples=1000):
    """Compute SHAP values for the given model."""
    try:
        import shap
    except ImportError:
        print("[Explainability] SHAP not installed. Using fallback.")
        return _fallback_importance(model, X_scaled, feature_names)

    n = X_scaled.shape[0]
    idx = np.random.RandomState(42).choice(n, min(max_samples, n), replace=False)
    X_sample = X_scaled[idx]
    print(f"[Explainability] Computing SHAP for {len(X_sample)} samples...")

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
    except Exception:
        try:
            bg = shap.sample(X_scaled, min(100, n))
            explainer = shap.KernelExplainer(model.decision_function, bg)
            shap_vals = explainer.shap_values(X_sample)
        except Exception as e:
            print(f"[Explainability] SHAP failed: {e}")
            return _fallback_importance(model, X_scaled, feature_names)

    imp = np.abs(shap_vals).mean(axis=0)
    imp_df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
    return {"shap_values": shap_vals, "feature_names": feature_names, "feature_importance": imp_df, "X_sample": X_sample, "method": "SHAP"}

def _fallback_importance(model, X_scaled, feature_names):
    """Permutation-based feature importance fallback."""
    print("[Explainability] Using permutation importance...")
    X_sub = X_scaled[:1000]
    base = model.decision_function(X_sub)
    imps = []
    for i in range(X_sub.shape[1]):
        Xp = X_sub.copy(); np.random.shuffle(Xp[:, i])
        imps.append(np.var(model.decision_function(Xp) - base))
    imps = np.array(imps)
    if imps.max() > 0: imps /= imps.max()
    df = pd.DataFrame({"feature": feature_names, "importance": imps}).sort_values("importance", ascending=False)
    return {"shap_values": None, "feature_names": feature_names, "feature_importance": df, "X_sample": X_sub, "method": "Permutation"}

def plot_feature_importance(result, output_path="data/visualizations/feature_importance.png", top_n=15):
    """Plot feature importance bar chart."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imp_df = result["feature_importance"].head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#1a1d29")
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
    ax.barh(range(len(imp_df)), imp_df["importance"].values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df["feature"].values, fontsize=11, color="#e0e0e0")
    ax.invert_yaxis()
    ax.set_title(f"Feature Importance ({result['method']})", fontsize=14, fontweight="bold", color="#00d4ff")
    ax.set_xlabel("Importance", color="#e0e0e0")
    ax.tick_params(colors="#b0b0b0")
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    print(f"[Explainability] Saved to {output_path}")
    return output_path

def plot_shap_summary(result, output_path="data/visualizations/shap_summary.png"):
    """Plot SHAP summary if available."""
    try:
        import shap
    except ImportError:
        return None
    if result["shap_values"] is None: return None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shap.summary_plot(result["shap_values"], features=result["X_sample"], feature_names=result["feature_names"], show=False)
    plt.savefig(output_path, bbox_inches="tight", dpi=150); plt.close()
    return output_path

def explain_transaction(row, feature_names, feature_importance, df_full=None):
    """Generate human-readable explanation for one transaction."""
    reasons = []
    if "amount" in row.index and df_full is not None and "amount" in df_full.columns:
        pct = (df_full["amount"] < row["amount"]).mean() * 100
        if pct > 99: reasons.append(f"Extremely high amount (${row['amount']:.2f}) — top 1%")
        elif pct > 95: reasons.append(f"High amount (${row['amount']:.2f}) — top 5%")
    if "hour" in row.index:
        h = row["hour"]
        if h <= 5 or h >= 23: reasons.append(f"Unusual time (hour: {h})")
    if row.get("error_present", 0) == 1: reasons.append("Transaction error detected")
    if "amount_zscore" in row.index and abs(row["amount_zscore"]) > 3:
        reasons.append(f"Amount {abs(row['amount_zscore']):.1f} sigma from user avg")
    if "time_gap_hours" in row.index and 0 < row["time_gap_hours"] < 0.05:
        reasons.append(f"Rapid transaction ({row['time_gap_hours']*60:.0f}m gap)")

    acols = [c for c in row.index if c.endswith("_anomaly") and c not in ("final_anomaly","risk_based_anomaly")]
    votes = sum(row.get(c, 0) for c in acols)
    if votes >= 3: reasons.append("Flagged by ALL models")
    elif votes == 2: reasons.append("Flagged by multiple models")
    elif votes == 1: reasons.append("Flagged by one model")

    if row.get("multimodal_score", 0) > 0.8: reasons.append(f"Critical multimodal score ({row['multimodal_score']:.3f})")
    elif row.get("multimodal_score", 0) > 0.6: reasons.append(f"High multimodal score ({row['multimodal_score']:.3f})")

    vlm = row.get("vlm_explanation", "")
    if vlm and vlm != "No VLM analysis available":
        reasons.append(f"AI Analysis: {vlm[:150]}...")

    if not reasons: reasons.append("Normal parameters")
    summary = " | ".join(r.split("—")[0].strip() if "—" in r else r for r in reasons[:5])
    return {"reasons": reasons, "summary": summary, "n_flags": len(reasons)}
