"""
One-Class SVM Anomaly Detector
================================
Learns a decision boundary around normal transactions.
Points outside the boundary are flagged as anomalies.
"""

import numpy as np
import joblib
from sklearn.svm import OneClassSVM


class OneClassSVMDetector:
    """One-Class SVM wrapper for fraud detection."""

    def __init__(
        self,
        kernel="rbf",
        gamma="scale",
        nu=0.01,
        max_samples=50000,
        random_state=42,
    ):
        self.model = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu,
        )
        self.max_samples = max_samples
        self.random_state = random_state
        self.name = "One-Class SVM"

    def fit(self, X_scaled):
        """
        Train One-Class SVM.
        Note: OC-SVM is O(n²) so we subsample for large datasets.
        """
        n_samples = X_scaled.shape[0]

        if n_samples > self.max_samples:
            print(
                f"[{self.name}] Subsampling {self.max_samples} from {n_samples} "
                f"(OC-SVM is O(n²))"
            )
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(n_samples, self.max_samples, replace=False)
            X_train = X_scaled[indices]
        else:
            X_train = X_scaled

        print(f"[{self.name}] Training on {X_train.shape[0]} samples...")
        self.model.fit(X_train)
        print(f"[{self.name}] Training complete.")
        return self

    def predict(self, X_scaled):
        """
        Return anomaly scores and binary predictions.
        decision_function: signed distance to boundary
        Negative = anomalous
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
