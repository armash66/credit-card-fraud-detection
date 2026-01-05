import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def train_isolation_forest(df_model):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    return model, scaler


def score_transactions(df_model, model, scaler):
    X_scaled = scaler.transform(df_model)
    scores = model.decision_function(X_scaled)
    flags = model.predict(X_scaled)

    return scores, flags
