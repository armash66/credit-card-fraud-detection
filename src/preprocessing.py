"""
Data Preprocessing & Feature Engineering Module
================================================
Handles all data cleaning, normalization, encoding,
and advanced feature engineering for the fraud detection pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings("ignore")


def load_raw_data(transactions_path, users_path=None, cards_path=None):
    """Load raw transaction data and optional user/card metadata."""
    df = pd.read_csv(transactions_path)
    print(f"[Preprocessing] Raw data loaded: {df.shape}")

    users = None
    cards = None

    if users_path:
        try:
            users = pd.read_csv(users_path)
            print(f"[Preprocessing] Users data loaded: {users.shape}")
        except Exception:
            pass

    if cards_path:
        try:
            cards = pd.read_csv(cards_path)
            print(f"[Preprocessing] Cards data loaded: {cards.shape}")
        except Exception:
            pass

    return df, users, cards


def clean_amount(df):
    """Clean and normalize the amount column."""
    df = df.copy()
    if df["amount"].dtype == object:
        df["amount"] = (
            df["amount"]
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
    return df


def generate_time_features(df):
    """Generate time-based features from the date column."""
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day_of_month"] = df["date"].dt.day
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(int)

        # Quarter of the day (0-3)
        df["day_quarter"] = df["hour"] // 6

    return df


def generate_error_features(df):
    """Generate features from the errors column."""
    df = df.copy()

    if "errors" in df.columns:
        df["error_present"] = df["errors"].notna().astype(int)
    elif "error_present" not in df.columns:
        df["error_present"] = 0

    return df


def generate_user_behavior_features(df):
    """
    Generate per-user behavioral features:
    - Transaction frequency
    - Average & std of amounts
    - Deviation from historical spending
    - Time gap between transactions
    - Merchant switching patterns
    """
    df = df.copy()

    # Identify the user column
    user_col = None
    for col in ["client_id", "cardholder_id", "user_id", "card_id"]:
        if col in df.columns:
            user_col = col
            break

    if user_col is None:
        print("[Preprocessing] No user column found — skipping user behavior features.")
        return df

    # --- Per-user statistics ---
    user_stats = df.groupby(user_col)["amount"].agg(
        user_avg_amount="mean",
        user_std_amount="std",
        user_max_amount="max",
        user_min_amount="min",
        user_txn_count="count",
    ).reset_index()

    user_stats["user_std_amount"] = user_stats["user_std_amount"].fillna(0)
    df = df.merge(user_stats, on=user_col, how="left")

    # Deviation from user's average
    df["amount_deviation"] = (df["amount"] - df["user_avg_amount"]).abs()
    df["amount_zscore"] = np.where(
        df["user_std_amount"] > 0,
        (df["amount"] - df["user_avg_amount"]) / df["user_std_amount"],
        0,
    )

    # --- Time gap between consecutive transactions ---
    if "date" in df.columns and df["date"].dtype != object:
        df = df.sort_values([user_col, "date"])
        df["time_gap_seconds"] = (
            df.groupby(user_col)["date"].diff().dt.total_seconds().fillna(0)
        )
        df["time_gap_hours"] = df["time_gap_seconds"] / 3600.0

    # --- Transaction frequency per hour ---
    if "hour" in df.columns:
        hour_freq = (
            df.groupby([user_col, "hour"])
            .size()
            .reset_index(name="hourly_txn_count")
        )
        df = df.merge(hour_freq, on=[user_col, "hour"], how="left")

    # --- Merchant switching ---
    if "mcc" in df.columns:
        df = df.sort_values([user_col, "date"] if "date" in df.columns else [user_col])
        df["merchant_switch"] = (
            df.groupby(user_col)["mcc"].shift(1) != df["mcc"]
        ).astype(int)

        # Unique merchants per user
        merchant_diversity = (
            df.groupby(user_col)["mcc"]
            .nunique()
            .reset_index(name="user_merchant_diversity")
        )
        df = df.merge(merchant_diversity, on=user_col, how="left")

    return df


def encode_categoricals(df, columns=None):
    """Encode categorical variables using one-hot or label encoding."""
    df = df.copy()

    if columns is None:
        columns = ["use_chip"]

    for col in columns:
        if col in df.columns:
            if df[col].nunique() <= 5:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
            else:
                le = LabelEncoder()
                df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
                df = df.drop(columns=[col])

    return df


def normalize_features(df, feature_cols, scaler=None):
    """Normalize numeric features using StandardScaler."""
    df = df.copy()

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_cols])
    else:
        X_scaled = scaler.transform(df[feature_cols])

    return X_scaled, scaler


def preprocess_pipeline(
    transactions_path,
    users_path=None,
    cards_path=None,
    sample_size=None,
    random_state=42,
):
    """
    Full preprocessing pipeline:
    1. Load data
    2. Clean amounts
    3. Generate time features
    4. Generate error features
    5. Generate user behavior features
    6. Encode categoricals
    7. Sample if needed
    """
    df, users, cards = load_raw_data(transactions_path, users_path, cards_path)

    # Store IDs
    if "id" in df.columns:
        df = df.rename(columns={"id": "transaction_id"})

    df = clean_amount(df)
    df = generate_time_features(df)
    df = generate_error_features(df)
    df = generate_user_behavior_features(df)
    df = encode_categoricals(df, columns=["use_chip"])

    # Sample if needed
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
        print(f"[Preprocessing] Sampled down to {len(df)} rows")

    print(f"[Preprocessing] Final shape: {df.shape}")
    print(f"[Preprocessing] Columns: {list(df.columns)}")

    return df


def get_feature_matrix(df):
    """Extract the feature matrix for model training."""
    # Core features
    feature_cols = [
        "amount",
        "hour",
        "day_of_week",
        "month",
        "mcc",
        "error_present",
    ]

    # Add engineered features if available
    optional_features = [
        "is_weekend",
        "is_night",
        "day_quarter",
        "day_of_month",
        "amount_deviation",
        "amount_zscore",
        "time_gap_hours",
        "hourly_txn_count",
        "merchant_switch",
        "user_merchant_diversity",
        "user_txn_count",
    ]

    # Add use_chip dummies
    chip_cols = [c for c in df.columns if c.startswith("use_chip_")]
    feature_cols.extend(chip_cols)

    for feat in optional_features:
        if feat in df.columns:
            feature_cols.append(feat)

    # Keep only available columns
    available = [c for c in feature_cols if c in df.columns]
    print(f"[Features] Using {len(available)} features: {available}")

    X = df[available].copy()
    X = X.fillna(0)

    return X, available
