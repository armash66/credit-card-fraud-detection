"""
Ensemble Anomaly Detector
===========================
Combines multiple anomaly detection models into a unified
risk score using weighted averaging and model agreement analysis.
"""

import numpy as np
import pandas as pd


class EnsembleDetector:
    """Combines anomaly scores from multiple models."""

    def __init__(
        self,
        weights=None,
        agreement_threshold=2,
        risk_percentile=0.99,
    ):
        """
        Args:
            weights: dict of model_name → weight.
                     Default: equal weights.
            agreement_threshold: minimum number of models that must
                                 agree for high-confidence flag.
            risk_percentile: percentile threshold for risk-based anomaly.
        """
        self.weights = weights
        self.agreement_threshold = agreement_threshold
        self.risk_percentile = risk_percentile
        self.name = "Ensemble"

    def combine(self, model_results):
        """
        Combine results from multiple models.

        Args:
            model_results: dict of model_name → {
                "anomaly_score": np.array [0,1],
                "is_anomaly": np.array {0,1},
            }

        Returns:
            DataFrame with ensemble scores and agreement analysis.
        """
        model_names = list(model_results.keys())
        n_models = len(model_names)

        # Default: equal weights
        if self.weights is None:
            self.weights = {name: 1.0 / n_models for name in model_names}

        # Ensure all models have equal-length outputs
        n_samples = len(next(iter(model_results.values()))["anomaly_score"])

        # Build result DataFrame
        df = pd.DataFrame()

        # Individual model scores
        total_weight = sum(self.weights.get(name, 1.0 / n_models) for name in model_names)
        weighted_sum = np.zeros(n_samples)

        for name in model_names:
            res = model_results[name]
            short_name = name.lower().replace(" ", "_").replace("-", "_")
            df[f"{short_name}_score"] = res["anomaly_score"]
            df[f"{short_name}_anomaly"] = res["is_anomaly"]

            w = self.weights.get(name, 1.0 / n_models)
            weighted_sum += w * res["anomaly_score"]

        # Ensemble risk score (weighted average)
        df["risk_score"] = weighted_sum / total_weight

        # Model agreement
        anomaly_cols = [c for c in df.columns if c.endswith("_anomaly")]
        df["model_votes"] = df[anomaly_cols].sum(axis=1)

        df["all_agree"] = (df["model_votes"] == n_models).astype(int)
        df["majority_agree"] = (
            df["model_votes"] >= self.agreement_threshold
        ).astype(int)

        # Risk-based anomaly
        risk_threshold = df["risk_score"].quantile(self.risk_percentile)
        df["risk_based_anomaly"] = (df["risk_score"] >= risk_threshold).astype(int)

        # Final anomaly: any model flags
        df["final_anomaly"] = (df["model_votes"] >= 1).astype(int)

        # Confidence level
        df["confidence"] = df.apply(
            lambda row: self._confidence_label(
                row["model_votes"],
                row["risk_score"],
                df["risk_score"].quantile(0.995),
                n_models,
            ),
            axis=1,
        )

        print(f"[Ensemble] Combined {n_models} models")
        print(f"  Risk threshold ({self.risk_percentile:.1%}): {risk_threshold:.4f}")
        print(f"  Anomaly rate: {df['final_anomaly'].mean():.2%}")
        print(f"  High-confidence rate: {df['all_agree'].mean():.4%}")

        return df

    @staticmethod
    def _confidence_label(votes, risk_score, high_threshold, n_models):
        """Assign confidence label based on model agreement and risk score."""
        if votes == n_models and risk_score >= high_threshold:
            return "Critical"
        elif votes == n_models:
            return "High"
        elif votes >= 2:
            return "Medium"
        elif votes == 1:
            return "Low"
        else:
            return "None"
