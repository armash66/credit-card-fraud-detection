import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

# ======================================================
# CONFIG
# ======================================================
MODEL_CHOICE = "both"
# options: "isolation_forest", "lof", "pca", "both"

SAMPLE_SIZE = 300_000
CONTAMINATION = 0.01
RANDOM_STATE = 42

# ======================================================
# 1. LOAD DATA
# ======================================================
df = pd.read_csv("credit card data/transactions_data.csv")
print("Initial shape:", df.shape)

# ======================================================
# 2. CLEAN & FEATURE ENGINEERING
# ======================================================
df["amount"] = df["amount"].str.replace("$", "", regex=False).astype(float)

df["date"] = pd.to_datetime(df["date"])
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

df["error_present"] = df["errors"].notna().astype(int)

# Keep transaction id
df_ids = df[["id"]].copy()
df = df.drop(columns=["errors", "date"])

# ======================================================
# 3. SAMPLE DATA
# ======================================================
df_sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
df_ids_sample = df_ids.loc[df_sample.index]

print("Sampled shape:", df_sample.shape)

# ======================================================
# 4. FEATURE MATRIX
# ======================================================
features = [
    "amount",
    "hour",
    "day_of_week",
    "month",
    "mcc",
    "error_present",
    "use_chip",
]

df_model = df_sample[features].copy()
df_model = pd.get_dummies(df_model, columns=["use_chip"], drop_first=True)

print("Model input shape:", df_model.shape)

# ======================================================
# 5. SCALE FEATURES
# ======================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model)

# ======================================================
# 6. ISOLATION FOREST
# ======================================================
iso = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
iso.fit(X_scaled)

df_sample["if_score"] = iso.decision_function(X_scaled)
df_sample["is_if_anomaly"] = (iso.predict(X_scaled) == -1).astype(int)

# ======================================================
# 7. LOCAL OUTLIER FACTOR
# ======================================================
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=CONTAMINATION,
    novelty=False,
)

lof_preds = lof.fit_predict(X_scaled)
df_sample["lof_score"] = -lof.negative_outlier_factor_
df_sample["is_lof_anomaly"] = (lof_preds == -1).astype(int)

print("\nLOF anomaly distribution:")
print(df_sample["is_lof_anomaly"].value_counts(normalize=True))

# ======================================================
# 8. PCA RECONSTRUCTION
# ======================================================
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_pca)

df_sample["pca_error"] = ((X_scaled - X_reconstructed) ** 2).mean(axis=1)

pca_threshold = np.percentile(df_sample["pca_error"], 99)
df_sample["is_pca_anomaly"] = (df_sample["pca_error"] > pca_threshold).astype(int)

print("\nPCA anomaly distribution:")
print(df_sample["is_pca_anomaly"].value_counts(normalize=True))

# ======================================================
# 9. MODEL AGREEMENT ANALYSIS
# ======================================================
df_sample["all_3_agree"] = (
    (df_sample["is_if_anomaly"] == 1)
    & (df_sample["is_lof_anomaly"] == 1)
    & (df_sample["is_pca_anomaly"] == 1)
).astype(int)

df_sample["any_2_agree"] = (
    (
        df_sample["is_if_anomaly"]
        + df_sample["is_lof_anomaly"]
        + df_sample["is_pca_anomaly"]
    )
    >= 2
).astype(int)

print("\nModel agreement rates:")
print("All 3 agree:", df_sample["all_3_agree"].mean())
print("At least 2 agree:", df_sample["any_2_agree"].mean())

# ======================================================
# 10. RISK SCORE
# ======================================================
def min_max(series):
    return (series - series.min()) / (series.max() - series.min())

df_sample["if_risk"] = min_max(-df_sample["if_score"])
df_sample["lof_risk"] = min_max(df_sample["lof_score"])
df_sample["pca_risk"] = min_max(df_sample["pca_error"])

df_sample["risk_score"] = (
    0.4 * df_sample["if_risk"]
    + 0.3 * df_sample["lof_risk"]
    + 0.3 * df_sample["pca_risk"]
)

risk_threshold = df_sample["risk_score"].quantile(0.99)
df_sample["risk_based_anomaly"] = (df_sample["risk_score"] >= risk_threshold).astype(int)

print("\nRisk-based anomaly rate:")
print(df_sample["risk_based_anomaly"].mean())

# ======================================================
# 11. FINAL MODEL SELECTION
# ======================================================
if MODEL_CHOICE == "isolation_forest":
    df_sample["final_anomaly"] = df_sample["is_if_anomaly"]

elif MODEL_CHOICE == "lof":
    df_sample["final_anomaly"] = df_sample["is_lof_anomaly"]

elif MODEL_CHOICE == "pca":
    df_sample["final_anomaly"] = df_sample["is_pca_anomaly"]

elif MODEL_CHOICE == "both":
    df_sample["final_anomaly"] = (
        (df_sample["is_if_anomaly"] == 1)
        | (df_sample["is_lof_anomaly"] == 1)
        | (df_sample["is_pca_anomaly"] == 1)
    ).astype(int)

else:
    raise ValueError("Invalid MODEL_CHOICE")

print("\nFinal anomaly rate:")
print(df_sample["final_anomaly"].value_counts(normalize=True))

# Attach transaction id LAST
df_sample["transaction_id"] = df_ids_sample["id"].values

# ======================================================
# 12. SAVE ARTIFACTS (MODELS + METADATA)
# ======================================================
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ---- save scored data for UI ----
df_sample.to_parquet(
    "data/scored_transactions.parquet",
    index=False
)

# ---- save reusable models ----
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(iso, "model/isolation_forest.pkl")
joblib.dump(pca, "model/pca.pkl")

# ---- save thresholds for consistent inference ----
thresholds = {
    "contamination": CONTAMINATION,
    "pca_threshold": float(pca_threshold),
    "risk_threshold": float(risk_threshold),
}

with open("model/thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

# ---- save run configuration ----
model_config = {
    "model_choice": MODEL_CHOICE,
    "sample_size": SAMPLE_SIZE,
    "random_state": RANDOM_STATE,
    "features": features,
}

with open("model/model_config.json", "w") as f:
    json.dump(model_config, f, indent=2)

print("✅ Models, thresholds, config, and scored data saved")

# ======================================================
# 13. OPTIONAL LABEL EVALUATION
# ======================================================
labels_path = "credit card data/train_fraud_labels.json"

if os.path.exists(labels_path):
    labels = pd.read_json(labels_path, lines=True)

    if "target" in labels.columns:
        labels = labels.rename(columns={"target": "is_fraud"})
    elif "is_fraud" not in labels.columns:
        labels = None

    if labels is not None:
        label_id_col = next(
            (c for c in labels.columns if c.lower() in ["transaction_id", "id"]),
            None,
        )

        if label_id_col:
            labels = labels.rename(columns={label_id_col: "transaction_id"})
            df_eval = df_sample.merge(labels, on="transaction_id", how="left")
            df_eval["is_fraud"] = df_eval["is_fraud"].fillna(0).astype(int)

            print("\nFraud distribution:")
            print(df_eval["is_fraud"].value_counts())
        else:
            print("ℹ️ Labels found but no transaction_id present → skipping evaluation.")
