"""
Isolation Forest Anomaly Detector
==================================
Detects globally rare and unusual transaction patterns
using tree-based isolation of anomalies.
"""

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """Isolation Forest wrapper for fraud detection."""

    def __init__(
        self,
        n_estimators=200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.name = "Isolation Forest"

    def fit(self, X_scaled):
        """Train the Isolation Forest model."""
        print(f"[{self.name}] Training on {X_scaled.shape[0]} samples...")
        self.model.fit(X_scaled)
        print(f"[{self.name}] Training complete.")
        return self

    def predict(self, X_scaled):
        """
        Return anomaly scores and binary predictions.
        Score: lower = more anomalous (decision_function)
        Prediction: -1 = anomaly, 1 = normal
        """
        scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        is_anomaly = (predictions == -1).astype(int)

        return {
            "raw_scores": scores,
            "is_anomaly": is_anomaly,
            "anomaly_score": self._normalize_scores(scores),
        }

    def _normalize_scores(self, scores):
        """Normalize scores to [0, 1] where 1 = most anomalous."""
        # Invert so higher = more anomalous
        inverted = -scores
        min_val = inverted.min()
        max_val = inverted.max()
        if max_val - min_val == 0:
            return np.zeros_like(inverted)
        return (inverted - min_val) / (max_val - min_val)

    def save(self, path):
        """Save model to disk."""
        joblib.dump(self.model, path)
        print(f"[{self.name}] Model saved to {path}")

    def load(self, path):
        """Load model from disk."""
        self.model = joblib.load(path)
        print(f"[{self.name}] Model loaded from {path}")
        return self
