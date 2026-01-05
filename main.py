import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
# ⚠️ Adjust path if needed
df = pd.read_csv("transactions_data.csv")

print("Initial shape:", df.shape)

# =========================
# 2. CLEAN AMOUNT COLUMN
# =========================
# Remove $ sign and convert to float
df["amount"] = (
    df["amount"]
    .str.replace("$", "", regex=False)
    .astype(float)
)

# =========================
# 3. PARSE DATE COLUMN
# =========================
df["date"] = pd.to_datetime(df["date"])

# Time-based features (VERY important for fraud)
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# =========================
# 4. ERROR FLAG FEATURE
# =========================
# Fraud often correlates with transaction errors
df["error_present"] = df["errors"].notna().astype(int)

# =========================
# 5. DROP USELESS COLUMNS
# =========================
df = df.drop(columns=["id", "errors", "date"])

# =========================
# 6. SAMPLE DATA (IMPORTANT)
# =========================
# Do NOT use full 13M rows initially
df_sample = df.sample(n=300_000, random_state=42)

print("Sampled shape:", df_sample.shape)

# =========================
# 7. QUICK CHECK
# =========================
print(df_sample.head())
print(df_sample.info())
