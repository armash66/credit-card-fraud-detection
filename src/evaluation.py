"""
Evaluation Module
==================
Metrics for evaluating the fraud detection system:
- Precision, Recall, F1-Score
- ROC-AUC
- False Positive Rate
- Baseline vs Multimodal comparison
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, average_precision_score,
)

def evaluate_model(y_true, y_pred, y_scores=None, model_name="Model"):
    """Evaluate a single model against ground truth labels."""
    results = {
        "model": model_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    results["true_positives"] = int(tp)
    results["false_positives"] = int(fp)
    results["true_negatives"] = int(tn)
    results["false_negatives"] = int(fn)
    results["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if y_scores is not None:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_scores)
            results["avg_precision"] = average_precision_score(y_true, y_scores)
        except Exception:
            results["roc_auc"] = 0.0
            results["avg_precision"] = 0.0

    print(f"\n[Evaluation] {model_name}:")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  FPR:       {results['fpr']:.4f}")
    if "roc_auc" in results:
        print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
    return results

def compare_models(y_true, model_predictions, model_scores=None):
    """Compare multiple models against ground truth."""
    all_results = []
    for name, y_pred in model_predictions.items():
        scores = model_scores.get(name) if model_scores else None
        res = evaluate_model(y_true, y_pred, scores, name)
        all_results.append(res)
    comparison_df = pd.DataFrame(all_results).set_index("model")
    print("\n[Evaluation] Model Comparison:")
    print(comparison_df.to_string())
    return comparison_df

def plot_roc_curves(y_true, model_scores, output_path="data/visualizations/roc_curves.png"):
    """Plot ROC curves for multiple models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#1a1d29")
    colors = ["#00d4ff", "#ff4444", "#ffa500", "#00ff88", "#ff66cc"]
    for i, (name, scores) in enumerate(model_scores.items()):
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2, label=f"{name} (AUC={auc:.3f})")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate", color="#e0e0e0", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="#e0e0e0", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", color="#00d4ff", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, facecolor="#1a1d29", edgecolor="#333", labelcolor="#e0e0e0")
    ax.tick_params(colors="#b0b0b0")
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    print(f"[Evaluation] ROC curves saved to {output_path}")
    return output_path

def plot_precision_recall_curves(y_true, model_scores, output_path="data/visualizations/pr_curves.png"):
    """Plot Precision-Recall curves."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#1a1d29")
    colors = ["#00d4ff", "#ff4444", "#ffa500", "#00ff88", "#ff66cc"]
    for i, (name, scores) in enumerate(model_scores.items()):
        try:
            prec, rec, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            ax.plot(rec, prec, color=colors[i % len(colors)], linewidth=2, label=f"{name} (AP={ap:.3f})")
        except Exception:
            pass
    ax.set_xlabel("Recall", color="#e0e0e0", fontsize=12)
    ax.set_ylabel("Precision", color="#e0e0e0", fontsize=12)
    ax.set_title("Precision-Recall Curves", color="#00d4ff", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, facecolor="#1a1d29", edgecolor="#333", labelcolor="#e0e0e0")
    ax.tick_params(colors="#b0b0b0")
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    return output_path

def generate_evaluation_report(comparison_df, output_path="data/evaluation_report.json"):
    """Save evaluation report as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = comparison_df.reset_index().to_dict(orient="records")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[Evaluation] Report saved to {output_path}")
    return output_path
