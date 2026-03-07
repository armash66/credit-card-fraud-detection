"""
Multimodal Fraud Detection Pipeline
=====================================
Orchestrates the full pipeline:
1. Data preprocessing & feature engineering
2. Multi-model unsupervised anomaly detection
3. Behavior visualization generation
4. Vision-Language Model analysis
5. Multimodal score fusion
6. Explainability (SHAP)
7. Evaluation (if labels available)
8. Save all artifacts

Usage:
    python pipeline.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import (
    preprocess_pipeline,
    get_feature_matrix,
    normalize_features,
)
from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.one_class_svm import OneClassSVMDetector
from src.models.lof import LOFDetector
from src.models.ensemble import EnsembleDetector
from src.visualization import (
    generate_batch_visualizations,
    generate_system_overview_charts,
)
from src.vlm_analyzer import (
    batch_analyze,
    get_cardholder_stats,
)
from src.multimodal_scorer import MultimodalScorer
from src.explainability import (
    compute_shap_values,
    plot_feature_importance,
    plot_shap_summary,
)
from src.evaluation import (
    evaluate_model,
    compare_models,
    plot_roc_curves,
    generate_evaluation_report,
)

# ======================================================
# CONFIGURATION
# ======================================================
CONFIG = {
    "data": {
        "transactions_path": "credit card data/transactions_data.csv",
        "users_path": "credit card data/users_data.csv",
        "cards_path": "cards_data.csv",
        "labels_path": "credit card data/train_fraud_labels.json",
    },
    "pipeline": {
        "sample_size": 300_000,
        "random_state": 42,
        "contamination": 0.01,
    },
    "models": {
        "isolation_forest": True,
        "autoencoder": True,
        "one_class_svm": True,
        "lof": True,
    },
    "ensemble": {
        "weights": {
            "Isolation Forest": 0.30,
            "Autoencoder": 0.30,
            "One-Class SVM": 0.20,
            "Local Outlier Factor": 0.20,
        },
    },
    "visualization": {
        "max_users": 50,
        "output_dir": "data/visualizations",
    },
    "vlm": {
        "force_simulated": True,  # Set False if GPU available
    },
    "multimodal": {
        "ml_weight": 0.6,
        "vlm_weight": 0.4,
    },
    "output": {
        "data_dir": "data",
        "model_dir": "model",
    },
}


def run_pipeline():
    """Execute the full multimodal fraud detection pipeline."""
    start_time = time.time()
    print("=" * 60)
    print("  MULTIMODAL FRAUD DETECTION PIPELINE")
    print("=" * 60)

    # =====================================================
    # STAGE 1: DATA PREPROCESSING
    # =====================================================
    print("\n📦 STAGE 1: Data Preprocessing & Feature Engineering")
    print("-" * 50)

    df = preprocess_pipeline(
        transactions_path=CONFIG["data"]["transactions_path"],
        users_path=CONFIG["data"]["users_path"],
        cards_path=CONFIG["data"]["cards_path"],
        sample_size=CONFIG["pipeline"]["sample_size"],
        random_state=CONFIG["pipeline"]["random_state"],
    )

    X, feature_names = get_feature_matrix(df)
    X_scaled, scaler = normalize_features(df, feature_names)

    # =====================================================
    # STAGE 2: MULTI-MODEL ANOMALY DETECTION
    # =====================================================
    print("\n🤖 STAGE 2: Multi-Model Anomaly Detection")
    print("-" * 50)

    model_results = {}

    # --- Isolation Forest ---
    if CONFIG["models"]["isolation_forest"]:
        iso = IsolationForestDetector(
            contamination=CONFIG["pipeline"]["contamination"],
            random_state=CONFIG["pipeline"]["random_state"],
        )
        iso.fit(X_scaled)
        model_results["Isolation Forest"] = iso.predict(X_scaled)

    # --- Autoencoder ---
    if CONFIG["models"]["autoencoder"]:
        ae = AutoencoderDetector(
            epochs=50,
            batch_size=256,
            contamination=CONFIG["pipeline"]["contamination"],
        )
        ae.fit(X_scaled)
        model_results["Autoencoder"] = ae.predict(X_scaled)

    # --- One-Class SVM ---
    if CONFIG["models"]["one_class_svm"]:
        ocsvm = OneClassSVMDetector(
            nu=CONFIG["pipeline"]["contamination"],
            max_samples=50000,
        )
        ocsvm.fit(X_scaled)
        model_results["One-Class SVM"] = ocsvm.predict(X_scaled)

    # --- Local Outlier Factor ---
    if CONFIG["models"]["lof"]:
        lof = LOFDetector(
            contamination=CONFIG["pipeline"]["contamination"],
            max_samples=100000,
        )
        lof.fit(X_scaled)
        lof_results = lof.predict()

        if lof_results.get("indices") is not None:
            # LOF was subsampled — expand to full dataset
            full_scores = np.zeros(len(X_scaled))
            full_anomaly = np.zeros(len(X_scaled), dtype=int)
            full_norm = np.zeros(len(X_scaled))
            idx = lof_results["indices"]
            full_scores[idx] = lof_results["raw_scores"]
            full_anomaly[idx] = lof_results["is_anomaly"]
            full_norm[idx] = lof_results["anomaly_score"]
            model_results["Local Outlier Factor"] = {
                "raw_scores": full_scores,
                "is_anomaly": full_anomaly,
                "anomaly_score": full_norm,
            }
        else:
            model_results["Local Outlier Factor"] = lof_results

    # --- Ensemble ---
    print("\n🔗 Ensemble Scoring")
    ensemble = EnsembleDetector(
        weights=CONFIG["ensemble"]["weights"],
    )
    ensemble_df = ensemble.combine(model_results)

    # Merge ensemble results into main df
    for col in ensemble_df.columns:
        df[col] = ensemble_df[col].values

    # =====================================================
    # STAGE 3: BEHAVIOR VISUALIZATION
    # =====================================================
    print("\n🎨 STAGE 3: Behavior Visualization")
    print("-" * 50)

    user_col = None
    for c in ["client_id", "cardholder_id", "user_id"]:
        if c in df.columns:
            user_col = c
            break

    viz_paths = {}
    if user_col:
        viz_paths = generate_batch_visualizations(
            df,
            output_dir=CONFIG["visualization"]["output_dir"],
            user_col=user_col,
            max_users=CONFIG["visualization"]["max_users"],
        )
        generate_system_overview_charts(
            df,
            output_dir=CONFIG["visualization"]["output_dir"],
        )
    else:
        print("[Pipeline] No user column found — skipping visualizations.")

    # =====================================================
    # STAGE 4: VLM ANALYSIS
    # =====================================================
    print("\n🧠 STAGE 4: Vision-Language Model Analysis")
    print("-" * 50)

    vlm_results = {}
    if viz_paths and user_col:
        # Compute per-user anomaly scores and stats
        user_scores = {}
        user_stats = {}
        for uid in viz_paths.keys():
            user_mask = df[user_col] == uid
            user_data = df[user_mask]
            if len(user_data) > 0 and "risk_score" in df.columns:
                user_scores[uid] = user_data["risk_score"].mean()
            user_stats[uid] = get_cardholder_stats(df, uid, user_col)

        vlm_results = batch_analyze(
            viz_paths,
            anomaly_scores=user_scores,
            all_stats=user_stats,
            force_simulated=CONFIG["vlm"]["force_simulated"],
        )
    else:
        print("[Pipeline] No visualizations — skipping VLM analysis.")

    # =====================================================
    # STAGE 5: MULTIMODAL SCORING
    # =====================================================
    print("\n⚡ STAGE 5: Multimodal Score Fusion")
    print("-" * 50)

    scorer = MultimodalScorer(
        ml_weight=CONFIG["multimodal"]["ml_weight"],
        vlm_weight=CONFIG["multimodal"]["vlm_weight"],
    )

    if vlm_results and user_col:
        df = scorer.compute_final_scores(
            df, vlm_results, user_col=user_col
        )
    else:
        df["multimodal_score"] = df.get("risk_score", 0)
        df["fraud_probability"] = df.get("risk_score", 0)
        df["vlm_suspicion_score"] = 0.0
        df["vlm_explanation"] = "VLM analysis not available"
        df["multimodal_risk_level"] = "⚪ N/A"

    # =====================================================
    # STAGE 6: EXPLAINABILITY
    # =====================================================
    print("\n🔍 STAGE 6: Explainability (SHAP)")
    print("-" * 50)

    if CONFIG["models"]["isolation_forest"]:
        shap_result = compute_shap_values(
            iso.model, X_scaled, feature_names
        )
        plot_feature_importance(shap_result)
        plot_shap_summary(shap_result)
    else:
        print("[Pipeline] IsolationForest not trained — skipping SHAP.")

    # =====================================================
    # STAGE 7: EVALUATION (if labels available)
    # =====================================================
    print("\n📊 STAGE 7: Evaluation")
    print("-" * 50)

    labels_path = CONFIG["data"]["labels_path"]
    if os.path.exists(labels_path):
        try:
            labels = pd.read_json(labels_path, lines=True)
            if "target" in labels.columns:
                labels = labels.rename(columns={"target": "is_fraud"})

            id_col = next(
                (c for c in labels.columns if c.lower() in ["transaction_id", "id"]),
                None,
            )
            if id_col and "transaction_id" in df.columns:
                labels = labels.rename(columns={id_col: "transaction_id"})
                df_eval = df.merge(labels, on="transaction_id", how="left")
                df_eval["is_fraud"] = df_eval["is_fraud"].fillna(0).astype(int)
                y_true = df_eval["is_fraud"].values

                if y_true.sum() > 0:
                    preds = {}
                    scores = {}
                    preds["Ensemble (Baseline)"] = df_eval["final_anomaly"].values
                    scores["Ensemble (Baseline)"] = df_eval["risk_score"].values

                    if "multimodal_score" in df_eval.columns:
                        preds["Multimodal"] = (df_eval["multimodal_score"] > 0.5).astype(int).values
                        scores["Multimodal"] = df_eval["multimodal_score"].values

                    comparison = compare_models(y_true, preds, scores)
                    plot_roc_curves(y_true, scores)
                    generate_evaluation_report(comparison)
                else:
                    print("[Evaluation] No fraud labels found in sample.")
            else:
                print("[Evaluation] Could not match label IDs.")
        except Exception as e:
            print(f"[Evaluation] Label loading failed: {e}")
    else:
        print("[Evaluation] No label file found — skipping.")

    # =====================================================
    # STAGE 8: SAVE ARTIFACTS
    # =====================================================
    print("\n💾 STAGE 8: Saving Artifacts")
    print("-" * 50)

    os.makedirs(CONFIG["output"]["data_dir"], exist_ok=True)
    os.makedirs(CONFIG["output"]["model_dir"], exist_ok=True)

    # Save scored data
    df.to_parquet("data/scored_transactions.parquet", index=False)
    print("  ✅ Scored transactions saved")

    # Save models
    joblib.dump(scaler, "model/scaler.pkl")
    if CONFIG["models"]["isolation_forest"]:
        iso.save("model/isolation_forest.pkl")
    if CONFIG["models"]["autoencoder"]:
        ae.save("model/autoencoder.pth")
    if CONFIG["models"]["one_class_svm"]:
        ocsvm.save("model/ocsvm.pkl")

    # Save config & thresholds
    with open("model/pipeline_config.json", "w") as f:
        json.dump(CONFIG, f, indent=2, default=str)

    # Save VLM results
    if vlm_results:
        vlm_save = {
            str(k): {
                "suspicion_score": v["suspicion_score"],
                "explanation": v["explanation"],
                "model": v.get("model", "Unknown"),
            }
            for k, v in vlm_results.items()
        }
        with open("data/vlm_results.json", "w") as f:
            json.dump(vlm_save, f, indent=2)
        print("  ✅ VLM results saved")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — {elapsed:.1f}s")
    print(f"  Transactions scored: {len(df):,}")
    print(f"  Anomaly rate: {df['final_anomaly'].mean():.2%}")
    if "multimodal_score" in df.columns:
        print(f"  Avg multimodal score: {df['multimodal_score'].mean():.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_pipeline()
