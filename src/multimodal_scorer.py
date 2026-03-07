"""
Multimodal Fraud Scorer
========================
Fuses anomaly detection scores from ML models with
Vision-Language Model suspicion scores to produce a
final multimodal fraud assessment.

Formula:
    final_score = (α × anomaly_score) + (β × vlm_suspicion_score)
    Default: α = 0.6, β = 0.4
"""

import numpy as np
import pandas as pd


class MultimodalScorer:
    """Combines ML anomaly scores with VLM suspicion scores."""

    def __init__(self, ml_weight=0.6, vlm_weight=0.4):
        """
        Args:
            ml_weight: weight for the ML anomaly score (default 0.6)
            vlm_weight: weight for the VLM suspicion score (default 0.4)
        """
        assert abs(ml_weight + vlm_weight - 1.0) < 1e-6, (
            "Weights must sum to 1.0"
        )
        self.ml_weight = ml_weight
        self.vlm_weight = vlm_weight
        self.name = "Multimodal Scorer"

    def compute_final_scores(
        self,
        df,
        vlm_results,
        user_col="client_id",
        ml_score_col="risk_score",
    ):
        """
        Compute final multimodal fraud scores.

        Args:
            df: DataFrame with ML anomaly scores (per-transaction)
            vlm_results: dict of cardholder_id → {suspicion_score, explanation}
            user_col: column identifying cardholders
            ml_score_col: column with ML-based risk score

        Returns:
            DataFrame with additional columns:
                - vlm_suspicion_score
                - vlm_explanation
                - multimodal_score
                - multimodal_risk_level
        """
        df = df.copy()

        # Map VLM scores to transactions via cardholder
        df["vlm_suspicion_score"] = 0.0
        df["vlm_explanation"] = "No VLM analysis available"
        df["vlm_model"] = "N/A"

        if user_col in df.columns:
            for uid, result in vlm_results.items():
                mask = df[user_col] == uid
                df.loc[mask, "vlm_suspicion_score"] = result["suspicion_score"]
                df.loc[mask, "vlm_explanation"] = result["explanation"]
                df.loc[mask, "vlm_model"] = result.get("model", "Unknown")

        # Compute multimodal score
        ml_scores = df[ml_score_col].values if ml_score_col in df.columns else np.zeros(len(df))
        vlm_scores = df["vlm_suspicion_score"].values

        df["multimodal_score"] = (
            self.ml_weight * ml_scores +
            self.vlm_weight * vlm_scores
        )

        # Risk levels
        df["multimodal_risk_level"] = df["multimodal_score"].apply(
            self._risk_level
        )

        # Fraud probability (calibrated sigmoid)
        df["fraud_probability"] = self._calibrate_probability(
            df["multimodal_score"].values
        )

        # Stats
        total = len(df)
        n_vlm_analyzed = (df["vlm_suspicion_score"] > 0).sum()
        avg_multimodal = df["multimodal_score"].mean()
        high_risk = (df["multimodal_score"] > 0.7).sum()

        print(f"[{self.name}] Multimodal scoring complete:")
        print(f"  Total transactions: {total:,}")
        print(f"  VLM-analyzed: {n_vlm_analyzed:,}")
        print(f"  Average multimodal score: {avg_multimodal:.4f}")
        print(f"  High-risk (>0.7): {high_risk:,}")

        return df

    @staticmethod
    def _risk_level(score):
        """Map multimodal score to human-readable risk level."""
        if score >= 0.8:
            return "Critical"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        else:
            return "Minimal"

    @staticmethod
    def _calibrate_probability(scores):
        """
        Convert raw multimodal scores to calibrated fraud probabilities
        using a stretched sigmoid function.
        """
        # Shift and scale for sigmoid calibration
        # Maps ~0.5 → ~0.5, with steeper transition near threshold
        k = 10  # steepness
        midpoint = 0.5
        probs = 1.0 / (1.0 + np.exp(-k * (scores - midpoint)))
        return np.clip(probs, 0.0, 1.0)

    def generate_report(self, df, top_n=20):
        """
        Generate a structured fraud report for the top-N riskiest transactions.

        Returns:
            list of report dicts
        """
        if "multimodal_score" not in df.columns:
            return []

        top_df = df.nlargest(top_n, "multimodal_score")

        reports = []
        for _, row in top_df.iterrows():
            report = {
                "transaction_id": row.get("transaction_id", "N/A"),
                "amount": row.get("amount", 0),
                "hour": row.get("hour", -1),
                "risk_score_ml": row.get("risk_score", 0),
                "vlm_suspicion_score": row.get("vlm_suspicion_score", 0),
                "multimodal_score": row.get("multimodal_score", 0),
                "fraud_probability": row.get("fraud_probability", 0),
                "risk_level": row.get("multimodal_risk_level", "Unknown"),
                "vlm_explanation": row.get("vlm_explanation", ""),
                "confidence": row.get("confidence", "Unknown"),
            }
            reports.append(report)

        return reports
