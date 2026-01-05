# 💳 Credit Card Fraud Detection System

An **end-to-end fraud detection system** built using **unsupervised machine learning** and an **analyst-focused interactive dashboard**.

This project is designed for **real-world scenarios** where fraud labels are scarce, delayed, or unavailable — a common challenge in financial systems.

---

## 🚀 Key Highlights

- 🔍 **Unsupervised Fraud Detection**
  - No reliance on labeled fraud data
  - Detects rare and evolving patterns

- 🤖 **Multi-Model Ensemble**
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - PCA Reconstruction Error

- ⚖️ **Risk-Based Scoring**
  - Combines multiple models into a single risk score
  - Prioritizes high-confidence alerts

- 🧠 **Explainable AI**
  - Clear, rule-based explanations for each flagged transaction
  - Answers the question: *“Why was this flagged?”*

- 📊 **Interactive Streamlit Dashboard**
  - Analyst-friendly UI
  - Filters, model agreement, confidence levels
  - CSV export for alerts

---

## 🧠 Detection Approach

### Models Used

| Model | Purpose |
|------|--------|
| **Isolation Forest** | Detects globally rare patterns |
| **Local Outlier Factor (LOF)** | Detects local density anomalies |
| **PCA Reconstruction Error** | Detects deviations from normal behavior |

Each model is **calibrated to flag ~1% anomalies** to allow **fair comparison and ensemble reasoning**.

---

### 🔗 Ensemble Logic

Instead of trusting a single model:
- We analyze **model agreement**
- Higher agreement ⇒ **higher confidence**
- A **weighted risk score** ranks transactions

This reduces false positives and improves interpretability.

---

## 🧭 Project Structure

credit-card-fraud-detection/
│
├── app/ # Streamlit application
│ ├── Home.py
│ └── pages/
│ ├── 1_Overview.py
│ ├── 2_Anomaly_Explorer.py
│ ├── 3_Behavior_Analysis.py
│ └── 4_Model_Insights.py
│
├── src/
│ └── data_loader.py # Centralized, safe data access
│
├── data/
│ └── scored_transactions.parquet
│
├── model/
│ ├── isolation_forest.pkl
│ └── scaler.pkl
│
├── main.py # Model training & scoring
├── requirements.txt
├── README.md
└── .gitignore

yaml
Copy code

---

## 🖥️ Dashboard Pages

### 🏠 Home
- Project overview
- Models used
- Navigation guide

### 📊 Overview
- System-wide statistics
- Anomaly rates
- Transaction patterns

### 🔍 Anomaly Explorer
- High-risk transactions ranked by risk score
- Human-readable explanations
- Model agreement filters
- CSV export

### 📊 Behavior Analysis
- Temporal fraud patterns
- Error vs anomaly behavior
- Amount vs risk visualization

### 🧠 Model Insights
- Individual model behavior
- Agreement analysis
- Confidence levels
- Risk score distribution

---

## 🛠️ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Run model pipeline
bash
Copy code
python main.py
3️⃣ Launch dashboard
bash
Copy code
streamlit run app/Home.py
🎯 Why This Project Matters
Mirrors real financial fraud systems

Focuses on interpretability, not just accuracy

Demonstrates ML + engineering + product thinking

Built with stability and scalability in mind

📌 Future Improvements
Add online / streaming anomaly detection

Adaptive thresholds per user or merchant

Semi-supervised learning when labels become available

👤 Author
Built as a portfolio-grade project to demonstrate applied machine learning, system design, and explainable AI.

yaml
Copy code

---