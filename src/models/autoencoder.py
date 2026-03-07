"""
Autoencoder Anomaly Detector (PyTorch)
=======================================
Deep learning-based anomaly detection using reconstruction error.
Transactions that are hard to reconstruct are flagged as anomalous.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TransactionAutoencoder(nn.Module):
    """
    Symmetric autoencoder architecture with bottleneck.
    Encoder: input → 64 → 32 → 16
    Decoder: 16 → 32 → 64 → input
    """

    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class AutoencoderDetector:
    """Autoencoder-based anomaly detector using reconstruction error."""

    def __init__(
        self,
        epochs=50,
        batch_size=256,
        learning_rate=1e-3,
        contamination=0.01,
        device=None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.contamination = contamination
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.threshold = None
        self.name = "Autoencoder"

    def fit(self, X_scaled):
        """Train the autoencoder on normal transaction data."""
        print(f"[{self.name}] Training on {X_scaled.shape[0]} samples (device: {self.device})...")

        input_dim = X_scaled.shape[1]
        self.model = TransactionAutoencoder(input_dim).to(self.device)

        # Create DataLoader
        tensor_x = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(tensor_x, tensor_x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs} — Loss: {avg_loss:.6f}")

        # Compute reconstruction errors and threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor_x)
            errors = torch.mean((tensor_x - reconstructed) ** 2, dim=1)
            errors = errors.cpu().numpy()

        self.threshold = np.percentile(errors, (1 - self.contamination) * 100)
        print(f"[{self.name}] Training complete. Threshold: {self.threshold:.6f}")

        return self

    def predict(self, X_scaled):
        """
        Compute reconstruction errors and return anomaly scores.
        Higher reconstruction error = more anomalous.
        """
        self.model.eval()
        tensor_x = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            reconstructed = self.model(tensor_x)
            errors = torch.mean((tensor_x - reconstructed) ** 2, dim=1)
            errors = errors.cpu().numpy()

        is_anomaly = (errors > self.threshold).astype(int)

        return {
            "raw_scores": errors,
            "is_anomaly": is_anomaly,
            "anomaly_score": self._normalize_scores(errors),
        }

    def _normalize_scores(self, scores):
        """Normalize scores to [0, 1] where 1 = most anomalous."""
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val == 0:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def save(self, path):
        """Save model weights and threshold."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "threshold": self.threshold,
                "input_dim": self.model.encoder[0].in_features,
            },
            path,
        )
        print(f"[{self.name}] Model saved to {path}")

    def load(self, path):
        """Load model weights and threshold."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        input_dim = checkpoint["input_dim"]
        self.model = TransactionAutoencoder(input_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.threshold = checkpoint["threshold"]
        print(f"[{self.name}] Model loaded from {path}")
        return self
