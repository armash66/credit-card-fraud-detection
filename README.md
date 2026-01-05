💳 Credit Card Fraud Detection System

An end-to-end fraud detection system built using unsupervised machine learning with an analyst-focused interactive dashboard.

This project targets real-world financial scenarios where fraud labels are scarce, delayed, or unavailable — a common challenge in production fraud systems.

🚀 Key Highlights
🔍 Unsupervised Fraud Detection
- No reliance on labeled fraud data
- Detects rare, novel, and evolving patterns

🤖 Multi-Model Ensemble
- Isolation Forest
- Local Outlier Factor (LOF)
- PCA Reconstruction Error

⚖️ Risk-Based Scoring
- Combines multiple model outputs into a single risk score
- Prioritizes high-confidence alerts

🧠 Explainable AI
- Rule-based, human-readable explanations
- Clearly answers: “Why was this transaction flagged?”

📊 Interactive Streamlit Dashboard
- Analyst-friendly UI
- Filters by risk, model agreement, and confidence
- CSV export for alerts

🧠 Detection Approach
🔬 Models Used
Model	Purpose
Isolation Forest	Detects globally rare patterns
Local Outlier Factor (LOF)	Detects local density anomalies
PCA Reconstruction Error	Detects deviations from normal behavior

Each model is calibrated to flag ~1% anomalies, ensuring:
- Fair comparison across models
- Meaningful ensemble reasoning

🔗 Ensemble Logic
Rather than trusting a single model:
- Analyze model agreement
- Higher agreement ⇒ higher confidence
- A weighted risk score ranks transactions

✅ Reduces false positives
✅ Improves interpretability for analysts

🧭 Project Structure
credit-card-fraud-detection/
│
├── app/                         # Streamlit application
│   ├── Home.py
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Anomaly_Explorer.py
│       ├── 3_Behavior_Analysis.py
│       └── 4_Model_Insights.py
│
├── src/
│   └── data_loader.py           # Centralized & safe data access
│
├── data/
│   └── scored_transactions.parquet
│
├── model/
│   ├── isolation_forest.pkl
│   └── scaler.pkl
│
├── main.py                      # Model training & scoring pipeline
├── requirements.txt
├── README.md
└── .gitignore

🖥️ Dashboard Pages
🏠 Home
- Project overview
- Models used
- Navigation guide

📊 Overview
System-wide statistics
Anomaly rates
Transaction behavior patterns

🔍 Anomaly Explorer
- High-risk transactions ranked by risk score
- Human-readable explanations
- Model agreement filters
- CSV export for analyst review

📈 Behavior Analysis
- Temporal anomaly patterns
- Error vs anomaly behavior
- Amount vs risk visualization

🧠 Model Insights
- Individual model behavior
- Model agreement analysis
- Confidence levels
- Risk score distribution

🛠️ How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Model Pipeline
python main.py

3️⃣ Launch Dashboard
streamlit run app/Home.py

🎯 Why This Project Matters
- Mirrors real-world financial fraud systems
- Designed for label-scarce environments
- Strong focus on interpretability over blind accuracy
- Demonstrates:
    - Machine Learning
    - System Design
    - Engineering Discipline
    - Product Thinking

📌 Future Improvements
- Online / streaming anomaly detection
- Adaptive thresholds per user or merchant
- Semi-supervised learning when labels become available

👤 Author
Built as a portfolio-grade project to demonstrate applied machine learning, explainable AI, and production-ready system design.