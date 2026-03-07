# Multimodal Credit Card Fraud Detection System

An advanced, research-grade fraud detection system combining **unsupervised machine learning**, **behavior visualization**, **Vision-Language Model reasoning**, and **explainable AI**.

---

## Key Highlights

### Multi-Model Anomaly Detection
- **Isolation Forest** — Detects globally rare transaction patterns
- **Autoencoder (PyTorch)** — Deep learning reconstruction error detection
- **One-Class SVM** — Decision boundary-based anomaly detection
- **Local Outlier Factor** — Density-based local anomaly detection
- **Weighted Ensemble** — Combines all models with configurable weights

### Behavior Visualization
- Spending heatmaps (hour x day of week)
- Transaction amount timelines with anomaly highlights
- Merchant category distributions
- Spending spike detection charts

### Vision-Language Model (VLM) Analysis
- Analyzes generated behavior visualizations using BLIP-2
- Produces natural language explanations of suspicious patterns
- Generates a suspicion score for multimodal fusion
- Simulated VLM fallback for environments without GPU

### Multimodal Score Fusion
```
final_score = 0.6 x ML_anomaly_score + 0.4 x VLM_suspicion_score
```
- Combines statistical anomaly detection with visual pattern reasoning
- Calibrated fraud probability output
- Tiered risk levels (Critical to Minimal)

### Explainable AI
- SHAP values for feature importance
- Permutation-based fallback when SHAP unavailable
- Per-transaction natural language explanations
- Model agreement analysis

### Interactive Dashboard
- 6-page Streamlit dashboard
- Fraud alerts with AI-generated explanations
- Interactive filters and drill-down
- CSV export for analyst review

---

## System Architecture

```
Raw Transaction Data
    |
Data Preprocessing & Feature Engineering
    |
+---------------------------------------+
|  Multi-Model Anomaly Detection        |
|  |-- Isolation Forest                 |
|  |-- Autoencoder (PyTorch)            |
|  |-- One-Class SVM                    |
|  +-- Local Outlier Factor             |
+---------------+-----------------------+
                |
        Ensemble Scoring
          |           |
  Behavior Viz -> VLM Analysis
          |           |
      Multimodal Score Fusion
                |
        SHAP Explainability
                |
        Streamlit Dashboard
```

---

## Project Structure

```
credit-card-fraud-detection/
|
|-- pipeline.py                      # Full multimodal pipeline orchestrator
|-- main.py                          # Original baseline pipeline (preserved)
|-- requirements.txt                 # Python dependencies
|-- README.md
|
|-- src/
|   |-- __init__.py
|   |-- data_loader.py               # Cached data access for dashboard
|   |-- preprocessing.py             # Data cleaning & feature engineering
|   |-- visualization.py             # Behavior visualization generator
|   |-- vlm_analyzer.py              # Vision-Language Model analysis
|   |-- multimodal_scorer.py         # Multimodal score fusion
|   |-- explainability.py            # SHAP & per-transaction explanations
|   |-- evaluation.py                # Precision/Recall/ROC-AUC metrics
|   +-- models/
|       |-- __init__.py
|       |-- isolation_forest.py      # Isolation Forest detector
|       |-- autoencoder.py           # PyTorch Autoencoder detector
|       |-- one_class_svm.py         # One-Class SVM detector
|       |-- lof.py                   # Local Outlier Factor detector
|       +-- ensemble.py             # Weighted ensemble combiner
|
|-- app/
|   |-- Home.py                      # Dashboard home page
|   +-- pages/
|       |-- 1_Overview.py            # System statistics
|       |-- 2_Anomaly_Explorer.py    # Transaction drill-down
|       |-- 3_Behavior_Analysis.py   # Temporal patterns
|       |-- 4_Model_Insights.py      # Model comparison
|       |-- 5_VLM_Analysis.py        # AI explanations
|       +-- 6_Explainability.py      # SHAP & evaluation
|
|-- data/
|   |-- scored_transactions.parquet  # Scored output data
|   |-- vlm_results.json             # VLM analysis results
|   |-- evaluation_report.json       # Eval metrics
|   +-- visualizations/              # Generated charts & images
|
|-- model/
|   |-- isolation_forest.pkl
|   |-- autoencoder.pth
|   |-- ocsvm.pkl
|   |-- scaler.pkl
|   +-- pipeline_config.json
|
+-- credit card data/
    |-- transactions_data.csv
    |-- train_fraud_labels.json
    |-- users_data.csv
    +-- mcc_codes.json
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Multimodal Pipeline
```bash
python pipeline.py
```

This runs all 8 stages:
1. Data preprocessing & feature engineering
2. Multi-model anomaly detection (4 models)
3. Behavior visualization generation
4. Vision-Language Model analysis
5. Multimodal score fusion
6. SHAP explainability
7. Evaluation (if labels available)
8. Save all artifacts

### 3. Launch the Dashboard
```bash
streamlit run app/Home.py
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Home** | System overview, models used, pipeline diagram |
| **Overview** | Transaction stats, amount distribution, hourly patterns |
| **Anomaly Explorer** | Ranked suspicious transactions with explanations |
| **Behavior Analysis** | Temporal anomaly patterns, error impact |
| **Model Insights** | Model agreement, confidence levels, risk distribution |
| **VLM Analysis** | AI-generated explanations with behavior visuals |
| **Explainability** | SHAP values, feature importance, evaluation metrics |

---

## Configuration

All pipeline parameters are configurable in `pipeline.py`:

```python
CONFIG = {
    "pipeline": {
        "sample_size": 300_000,
        "contamination": 0.01,
    },
    "models": {
        "isolation_forest": True,
        "autoencoder": True,
        "one_class_svm": True,
        "lof": True,
    },
    "ensemble": {
        "weights": {
            "Isolation Forest": 0.30,
            "Autoencoder": 0.30,
            "One-Class SVM": 0.20,
            "Local Outlier Factor": 0.20,
        },
    },
    "multimodal": {
        "ml_weight": 0.6,    # Weight for ML score
        "vlm_weight": 0.4,   # Weight for VLM score
    },
    "vlm": {
        "force_simulated": True,  # Set False for GPU with BLIP-2
    },
}
```

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| ML Models | Scikit-learn, PyTorch |
| VLM | HuggingFace Transformers (BLIP-2) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Data | Pandas, NumPy, PyArrow |

---

## Evaluation Metrics

When fraud labels are available, the system evaluates using:
- **Precision** — How many flagged transactions are truly fraudulent
- **Recall** — How many fraudulent transactions are caught
- **F1-Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Area under the ROC curve
- **False Positive Rate** — Rate of false alarms

Compares **baseline (ML-only)** vs **multimodal (ML + VLM)** performance.

---

## Future Improvements
- Online / streaming anomaly detection
- Adaptive thresholds per user or merchant
- Semi-supervised learning when labels become available
- Real BLIP-2 / LLaVA integration with GPU
- Geographic distance features between transactions
- Graph-based fraud detection (transaction networks)

---

## Author
Built as a **research-grade portfolio project** demonstrating:
- Advanced Machine Learning (multi-model ensemble)
- Deep Learning (PyTorch Autoencoder)
- Multimodal AI (Vision-Language Models)
- Explainable AI (SHAP)
- System Design & Engineering
- Interactive Data Visualization