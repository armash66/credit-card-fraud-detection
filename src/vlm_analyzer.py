"""
Vision-Language Model (VLM) Analyzer
======================================
Uses a Vision-Language Model to analyze transaction behavior
visualizations and generate natural language explanations
of suspicious patterns.

Supports:
- BLIP-2 (primary, from HuggingFace transformers)
- Simulated VLM (fallback for environments without GPU)

The VLM analyzes generated behavior images and produces:
- A suspicion score (0-1)
- A natural language explanation
"""

import os
import re
import numpy as np

# Lazy imports to avoid loading heavy models at import time
_vlm_model = None
_vlm_processor = None
_vlm_device = None
_vlm_mode = None  # "blip2" or "simulated"


def _load_vlm(force_simulated=False):
    """Lazily load the VLM model."""
    global _vlm_model, _vlm_processor, _vlm_device, _vlm_mode

    if _vlm_model is not None:
        return

    if force_simulated:
        _vlm_mode = "simulated"
        _vlm_model = "simulated"
        print("[VLM] Using simulated VLM mode (no GPU required)")
        return

    try:
        import torch
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        _vlm_device = "cuda" if torch.cuda.is_available() else "cpu"

        if _vlm_device == "cpu":
            print("[VLM] No GPU detected. Falling back to simulated VLM.")
            _vlm_mode = "simulated"
            _vlm_model = "simulated"
            return

        print("[VLM] Loading BLIP-2 model (this may take a minute)...")
        model_name = "Salesforce/blip2-opt-2.7b"

        _vlm_processor = Blip2Processor.from_pretrained(model_name)
        _vlm_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(_vlm_device)

        _vlm_mode = "blip2"
        print(f"[VLM] BLIP-2 loaded on {_vlm_device}")

    except Exception as e:
        print(f"[VLM] Failed to load BLIP-2: {e}")
        print("[VLM] Falling back to simulated VLM mode.")
        _vlm_mode = "simulated"
        _vlm_model = "simulated"


def analyze_behavior_image(
    image_path,
    anomaly_score=None,
    transaction_stats=None,
    force_simulated=False,
):
    """
    Analyze a behavior visualization image using VLM.

    Args:
        image_path: path to the behavior visualization png
        anomaly_score: optional ML anomaly score for context
        transaction_stats: optional dict of transaction statistics

    Returns:
        dict with:
            - suspicion_score: float [0, 1]
            - explanation: str
            - raw_response: str
    """
    _load_vlm(force_simulated=force_simulated)

    if _vlm_mode == "blip2":
        return _analyze_with_blip2(image_path, anomaly_score, transaction_stats)
    else:
        return _analyze_simulated(image_path, anomaly_score, transaction_stats)


def _analyze_with_blip2(image_path, anomaly_score, transaction_stats):
    """Analyze using real BLIP-2 model."""
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    prompt = (
        "Analyze this transaction behavior visualization. "
        "Identify whether the pattern indicates fraudulent activity. "
        "Look for unusual spending spikes, irregular timing patterns, "
        "concentrated transactions, or abnormal merchant switching. "
        "Explain your reasoning."
    )

    inputs = _vlm_processor(images=image, text=prompt, return_tensors="pt").to(
        _vlm_device, torch.float16
    )

    generated_ids = _vlm_model.generate(
        **inputs,
        max_new_tokens=200,
        num_beams=3,
        temperature=0.7,
    )

    raw_response = _vlm_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    # Extract suspicion score from response
    suspicion_score = _extract_suspicion_score(raw_response, anomaly_score)

    return {
        "suspicion_score": suspicion_score,
        "explanation": raw_response,
        "raw_response": raw_response,
        "model": "BLIP-2",
    }


def _analyze_simulated(image_path, anomaly_score, transaction_stats):
    """
    Simulated VLM analysis for development environments without GPU.
    Uses heuristic analysis of transaction statistics to generate
    human-readable explanations that mimic VLM output.
    """
    suspicion_indicators = []
    suspicion_score = 0.0

    if transaction_stats is None:
        transaction_stats = {}

    # --- Analyze transaction statistics ---

    # High anomaly score from ML models
    if anomaly_score is not None:
        if anomaly_score > 0.8:
            suspicion_indicators.append(
                "The ML models indicate a very high anomaly score, "
                "suggesting this cardholder's behavior deviates significantly "
                "from normal patterns."
            )
            suspicion_score += 0.35
        elif anomaly_score > 0.5:
            suspicion_indicators.append(
                "The ML anomaly score is moderately elevated, "
                "indicating some deviation from typical behavior."
            )
            suspicion_score += 0.2

    # Amount deviation
    avg_amount = transaction_stats.get("avg_amount", 0)
    max_amount = transaction_stats.get("max_amount", 0)
    std_amount = transaction_stats.get("std_amount", 0)

    if max_amount > 0 and avg_amount > 0:
        spike_ratio = max_amount / avg_amount
        if spike_ratio > 10:
            suspicion_indicators.append(
                f"Significant spending spikes detected — maximum transaction "
                f"(${max_amount:.0f}) is {spike_ratio:.0f}x the average "
                f"(${avg_amount:.0f}), indicating potential card compromise."
            )
            suspicion_score += 0.25
        elif spike_ratio > 5:
            suspicion_indicators.append(
                f"Moderate spending spikes observed — peak amount is "
                f"{spike_ratio:.0f}x the average spending."
            )
            suspicion_score += 0.15

    # High-variance spending
    if std_amount > 0 and avg_amount > 0:
        cv = std_amount / avg_amount
        if cv > 2.0:
            suspicion_indicators.append(
                "Very high variance in transaction amounts indicates "
                "irregular spending behavior."
            )
            suspicion_score += 0.1

    # Night transactions
    night_ratio = transaction_stats.get("night_txn_ratio", 0)
    if night_ratio > 0.3:
        suspicion_indicators.append(
            f"{night_ratio:.0%} of transactions occur during unusual hours "
            f"(11PM–5AM), which is a common indicator of fraudulent activity."
        )
        suspicion_score += 0.15

    # Merchant diversity
    merchant_diversity = transaction_stats.get("merchant_diversity", 0)
    txn_count = transaction_stats.get("txn_count", 0)
    if merchant_diversity > 0 and txn_count > 0:
        diversity_ratio = merchant_diversity / txn_count
        if diversity_ratio > 0.8:
            suspicion_indicators.append(
                "High merchant diversity relative to transaction volume — "
                "each transaction is at a different merchant, which may indicate "
                "card testing behavior."
            )
            suspicion_score += 0.15

    # Error rate
    error_rate = transaction_stats.get("error_rate", 0)
    if error_rate > 0.1:
        suspicion_indicators.append(
            f"Elevated transaction error rate ({error_rate:.0%}) suggests "
            f"multiple failed or rejected transaction attempts."
        )
        suspicion_score += 0.1

    # Clamp score
    suspicion_score = min(1.0, max(0.0, suspicion_score))

    # Generate explanation
    if not suspicion_indicators:
        explanation = (
            "Based on the transaction behavior analysis, this cardholder shows "
            "normal spending patterns. No significant anomalies or suspicious "
            "indicators were detected in the spending heatmap, transaction "
            "timeline, or merchant distribution."
        )
    else:
        explanation = "Transaction behavior analysis reveals the following concerns:\n\n"
        for i, indicator in enumerate(suspicion_indicators, 1):
            explanation += f"{i}. {indicator}\n\n"

        if suspicion_score > 0.6:
            explanation += (
                "Overall assessment: HIGH RISK — Multiple strong indicators of "
                "potentially fraudulent activity. Recommend immediate review."
            )
        elif suspicion_score > 0.3:
            explanation += (
                "Overall assessment: MODERATE RISK — Some suspicious patterns "
                "detected that warrant further investigation."
            )
        else:
            explanation += (
                "Overall assessment: LOW RISK — Minor anomalies detected, "
                "but likely within normal behavioral variation."
            )

    return {
        "suspicion_score": suspicion_score,
        "explanation": explanation,
        "raw_response": explanation,
        "model": "Simulated VLM (heuristic)",
    }


def _extract_suspicion_score(text, anomaly_score=None):
    """
    Extract or estimate a suspicion score from VLM text output.
    Falls back to anomaly_score with noise if extraction fails.
    """
    text_lower = text.lower()

    # Try to find explicit score
    score_patterns = [
        r"suspicion\s*(?:score|level)[:\s]*(\d*\.?\d+)",
        r"risk\s*(?:score|level)[:\s]*(\d*\.?\d+)",
        r"(\d*\.?\d+)\s*(?:out of|/)\s*(?:1|10|100)",
    ]

    for pattern in score_patterns:
        match = re.search(pattern, text_lower)
        if match:
            score = float(match.group(1))
            if score > 1:
                score = score / 100 if score <= 100 else score / 1000
            return min(1.0, max(0.0, score))

    # Keyword-based estimation
    high_keywords = ["fraud", "suspicious", "abnormal", "irregular", "concerning"]
    low_keywords = ["normal", "typical", "expected", "regular", "legitimate"]

    high_count = sum(1 for k in high_keywords if k in text_lower)
    low_count = sum(1 for k in low_keywords if k in text_lower)

    if high_count > low_count:
        base = 0.6 + (high_count - low_count) * 0.1
    elif low_count > high_count:
        base = 0.3 - (low_count - high_count) * 0.05
    else:
        base = 0.5

    # Blend with anomaly score if available
    if anomaly_score is not None:
        base = 0.5 * base + 0.5 * anomaly_score

    return min(1.0, max(0.0, base))


def batch_analyze(
    visualization_paths,
    anomaly_scores=None,
    all_stats=None,
    force_simulated=False,
):
    """
    Analyze multiple cardholder visualizations.

    Args:
        visualization_paths: dict of cardholder_id → image_path
        anomaly_scores: dict of cardholder_id → ml_anomaly_score
        all_stats: dict of cardholder_id → transaction_stats
        force_simulated: force simulated VLM mode

    Returns:
        dict of cardholder_id → analysis_result
    """
    _load_vlm(force_simulated=force_simulated)

    results = {}
    n_total = len(visualization_paths)

    print(f"[VLM] Analyzing {n_total} visualizations (mode: {_vlm_mode})...")

    for i, (uid, path) in enumerate(visualization_paths.items()):
        if not os.path.exists(path):
            continue

        score = anomaly_scores.get(uid, None) if anomaly_scores else None
        stats = all_stats.get(uid, {}) if all_stats else {}

        results[uid] = analyze_behavior_image(
            path,
            anomaly_score=score,
            transaction_stats=stats,
            force_simulated=force_simulated,
        )

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{n_total}")

    print(f"[VLM] Analysis complete. {len(results)} results generated.")
    return results


def get_cardholder_stats(df, cardholder_id, user_col="client_id"):
    """Compute transaction statistics for a cardholder (for VLM context)."""
    user_df = df[df[user_col] == cardholder_id]

    if len(user_df) == 0:
        return {}

    stats = {
        "txn_count": len(user_df),
        "avg_amount": user_df["amount"].mean(),
        "std_amount": user_df["amount"].std(),
        "max_amount": user_df["amount"].max(),
        "min_amount": user_df["amount"].min(),
        "total_amount": user_df["amount"].sum(),
    }

    if "is_night" in user_df.columns:
        stats["night_txn_ratio"] = user_df["is_night"].mean()

    if "mcc" in user_df.columns:
        stats["merchant_diversity"] = user_df["mcc"].nunique()

    if "error_present" in user_df.columns:
        stats["error_rate"] = user_df["error_present"].mean()

    if "final_anomaly" in user_df.columns:
        stats["anomaly_count"] = user_df["final_anomaly"].sum()
        stats["anomaly_rate"] = user_df["final_anomaly"].mean()

    return stats
