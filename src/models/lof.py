"""
Local Outlier Factor (LOF) Anomaly Detector
=============================================
Detects local density-based anomalies — transactions that are
isolated relative to their neighbors, rather than globally rare.
"""

import numpy as np
import joblib
from sklearn.neighbors import LocalOutlierFactor


class LOFDetector:
    """LOF wrapper for fraud detection."""

    def __init__(
        self,
        n_neighbors=20,
        contamination=0.01,
        max_samples=100000,
        random_state=42,
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.name = "Local Outlier Factor"

        # LOF with novelty=False for fit_predict mode
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False,
        )

    def fit(self, X_scaled):
        """
        LOF fit_predict mode — fits and predicts in one step.
        Stores results for later retrieval.
        """
        n_samples = X_scaled.shape[0]

        if n_samples > self.max_samples:
            print(
                f"[{self.name}] Subsampling {self.max_samples} from {n_samples} "
                f"for performance"
            )
            rng = np.random.RandomState(self.random_state)
            self._indices = rng.choice(n_samples, self.max_samples, replace=False)
            X_train = X_scaled[self._indices]
        else:
            self._indices = None
            X_train = X_scaled

        print(f"[{self.name}] Training on {X_train.shape[0]} samples...")

        predictions = self.model.fit_predict(X_train)
        self._lof_scores = -self.model.negative_outlier_factor_
        self._predictions = predictions
        self._X_train = X_train

        print(f"[{self.name}] Training complete.")
        return self

    def predict(self, X_scaled=None):
        """
        Return anomaly scores.
        Note: LOF with novelty=False cannot predict on new data,
        so we return stored results from fit.
        """
        scores = self._lof_scores
        is_anomaly = (self._predictions == -1).astype(int)

        return {
            "raw_scores": scores,
            "is_anomaly": is_anomaly,
            "anomaly_score": self._normalize_scores(scores),
            "indices": self._indices,
        }

    def _normalize_scores(self, scores):
        """Normalize scores to [0, 1] where 1 = most anomalous."""
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val == 0:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)
