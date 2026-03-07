"""
Behavior Visualization Module
===============================
Generates visual representations of cardholder transaction
behavior for VLM analysis and dashboard display.

Produces:
- Spending heatmaps (hour × day_of_week)
- Transaction timelines
- Merchant category distributions
- Spending spike detection charts
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Publication-quality style
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#1a1d29",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#b0b0b0",
    "ytick.color": "#b0b0b0",
    "axes.edgecolor": "#333333",
    "grid.color": "#2a2d3a",
    "figure.dpi": 150,
    "font.size": 10,
})


def generate_cardholder_visualization(
    df,
    cardholder_id,
    output_dir="data/visualizations",
    user_col="client_id",
):
    """
    Generate a comprehensive behavior visualization for a single cardholder.

    Creates a 2×2 subplot figure:
    1. Spending heatmap (hour × day_of_week)
    2. Transaction amount timeline
    3. Merchant category distribution
    4. Spending pattern with anomaly highlights

    Returns: path to saved image
    """
    os.makedirs(output_dir, exist_ok=True)

    user_df = df[df[user_col] == cardholder_id].copy()

    if len(user_df) < 3:
        return None

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    fig.suptitle(
        f"Transaction Behavior — Cardholder {cardholder_id}",
        fontsize=16,
        fontweight="bold",
        color="#00d4ff",
        y=0.98,
    )

    # --- 1. Spending Heatmap (Hour × Day of Week) ---
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_spending_heatmap(ax1, user_df)

    # --- 2. Transaction Timeline ---
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_transaction_timeline(ax2, user_df)

    # --- 3. Merchant Category Distribution ---
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_merchant_distribution(ax3, user_df)

    # --- 4. Spending Spike Detection ---
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_spending_spikes(ax4, user_df)

    # Save
    filename = f"cardholder_{cardholder_id}_behavior.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return filepath


def _plot_spending_heatmap(ax, user_df):
    """Plot spending heatmap: hour × day_of_week colored by total amount."""
    if "hour" not in user_df.columns or "day_of_week" not in user_df.columns:
        ax.text(
            0.5, 0.5, "Insufficient time data",
            ha="center", va="center", fontsize=12, color="#888"
        )
        ax.set_title("Spending Heatmap", fontsize=12, color="#00d4ff")
        return

    pivot = user_df.pivot_table(
        values="amount",
        index="day_of_week",
        columns="hour",
        aggfunc="sum",
        fill_value=0,
    )

    # Ensure all hours and days are present
    all_hours = range(24)
    all_days = range(7)
    pivot = pivot.reindex(index=all_days, columns=all_hours, fill_value=0)

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="#1a1d29",
        cbar_kws={"label": "Total Amount ($)", "shrink": 0.8},
        yticklabels=day_labels,
    )
    ax.set_title("Spending Heatmap (Hour × Day)", fontsize=12, color="#00d4ff")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")


def _plot_transaction_timeline(ax, user_df):
    """Plot transaction amounts over time."""
    if "date" in user_df.columns:
        sorted_df = user_df.sort_values("date")
        x_vals = range(len(sorted_df))
    else:
        sorted_df = user_df
        x_vals = range(len(sorted_df))

    amounts = sorted_df["amount"].values

    # Color by anomaly status if available
    if "final_anomaly" in sorted_df.columns:
        colors = [
            "#ff4444" if a == 1 else "#00d4ff"
            for a in sorted_df["final_anomaly"].values
        ]
    else:
        colors = ["#00d4ff"] * len(amounts)

    ax.bar(x_vals, amounts, color=colors, alpha=0.8, width=1.0)

    # Average line
    avg_amount = amounts.mean()
    ax.axhline(y=avg_amount, color="#ffa500", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Avg: ${avg_amount:.0f}")

    # 95th percentile line
    p95 = np.percentile(amounts, 95)
    ax.axhline(y=p95, color="#ff4444", linestyle=":", linewidth=1.5, alpha=0.7, label=f"P95: ${p95:.0f}")

    ax.set_title("Transaction Amount Timeline", fontsize=12, color="#00d4ff")
    ax.set_xlabel("Transaction #")
    ax.set_ylabel("Amount ($)")
    ax.legend(fontsize=8, loc="upper right")


def _plot_merchant_distribution(ax, user_df):
    """Plot merchant category distribution."""
    if "mcc" not in user_df.columns:
        ax.text(
            0.5, 0.5, "No merchant data",
            ha="center", va="center", fontsize=12, color="#888"
        )
        ax.set_title("Merchant Distribution", fontsize=12, color="#00d4ff")
        return

    top_merchants = user_df["mcc"].value_counts().head(10)

    bars = ax.barh(
        range(len(top_merchants)),
        top_merchants.values,
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_merchants))),
        alpha=0.85,
    )
    ax.set_yticks(range(len(top_merchants)))
    ax.set_yticklabels([f"MCC {m}" for m in top_merchants.index], fontsize=9)
    ax.set_title("Top Merchant Categories", fontsize=12, color="#00d4ff")
    ax.set_xlabel("Transaction Count")
    ax.invert_yaxis()


def _plot_spending_spikes(ax, user_df):
    """Plot amount distribution with spike detection."""
    amounts = user_df["amount"].values

    if len(amounts) < 5:
        ax.text(
            0.5, 0.5, "Insufficient data",
            ha="center", va="center", fontsize=12, color="#888"
        )
        return

    # Histogram
    ax.hist(
        amounts,
        bins=min(30, len(amounts) // 2 + 1),
        color="#00d4ff",
        alpha=0.7,
        edgecolor="#1a1d29",
    )

    # Mark anomalous amounts
    mean_amount = amounts.mean()
    std_amount = amounts.std()
    spike_threshold = mean_amount + 2 * std_amount

    if std_amount > 0:
        spikes = amounts[amounts > spike_threshold]
        if len(spikes) > 0:
            ax.axvline(
                x=spike_threshold,
                color="#ff4444",
                linestyle="--",
                linewidth=1.5,
                label=f"Spike threshold: ${spike_threshold:.0f}",
            )

    ax.set_title("Spending Distribution & Spikes", fontsize=12, color="#00d4ff")
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8, loc="upper right")


def generate_batch_visualizations(
    df,
    output_dir="data/visualizations",
    user_col="client_id",
    max_users=50,
    priority="anomalous",
):
    """
    Generate visualizations for multiple cardholders.

    Args:
        df: scored DataFrame
        output_dir: where to save images
        user_col: column identifying cardholders
        max_users: maximum number of users to visualize
        priority: "anomalous" to prioritize users with anomalies,
                  "random" for random selection

    Returns:
        dict of cardholder_id → file_path
    """
    if user_col not in df.columns:
        print(f"[Visualization] User column '{user_col}' not found. Skipping.")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    # Select users to visualize
    if priority == "anomalous" and "final_anomaly" in df.columns:
        # Prioritize users with most anomalies
        user_anomaly_count = (
            df.groupby(user_col)["final_anomaly"]
            .sum()
            .sort_values(ascending=False)
        )
        selected_users = user_anomaly_count.head(max_users).index.tolist()
    else:
        unique_users = df[user_col].unique()
        rng = np.random.RandomState(42)
        selected_users = rng.choice(
            unique_users,
            min(max_users, len(unique_users)),
            replace=False,
        ).tolist()

    print(f"[Visualization] Generating visuals for {len(selected_users)} cardholders...")

    results = {}
    for i, uid in enumerate(selected_users):
        path = generate_cardholder_visualization(
            df, uid, output_dir, user_col
        )
        if path:
            results[uid] = path

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(selected_users)}")

    print(f"[Visualization] Generated {len(results)} visualizations.")
    return results


def generate_system_overview_charts(df, output_dir="data/visualizations"):
    """Generate system-level overview charts for the dashboard."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "System-Wide Fraud Detection Overview",
        fontsize=16,
        fontweight="bold",
        color="#00d4ff",
        y=0.98,
    )

    # 1. Risk score distribution
    if "risk_score" in df.columns:
        axes[0, 0].hist(
            df["risk_score"], bins=100, color="#00d4ff", alpha=0.8, edgecolor="#1a1d29"
        )
        axes[0, 0].set_title("Risk Score Distribution", color="#00d4ff")
        axes[0, 0].set_xlabel("Risk Score")
        axes[0, 0].set_ylabel("Count")

    # 2. Anomalies by hour
    if "hour" in df.columns and "final_anomaly" in df.columns:
        anomalies = df[df["final_anomaly"] == 1]
        if not anomalies.empty:
            hourly = anomalies["hour"].value_counts().sort_index()
            axes[0, 1].bar(hourly.index, hourly.values, color="#ff4444", alpha=0.8)
            axes[0, 1].set_title("Anomalies by Hour", color="#00d4ff")
            axes[0, 1].set_xlabel("Hour")
            axes[0, 1].set_ylabel("Count")

    # 3. Amount distribution: normal vs anomalous
    if "final_anomaly" in df.columns:
        normal = df[df["final_anomaly"] == 0]["amount"]
        anomalous = df[df["final_anomaly"] == 1]["amount"]

        axes[1, 0].hist(
            normal.clip(upper=500), bins=50, alpha=0.6,
            color="#00d4ff", label="Normal", edgecolor="#1a1d29"
        )
        if not anomalous.empty:
            axes[1, 0].hist(
                anomalous.clip(upper=500), bins=50, alpha=0.6,
                color="#ff4444", label="Anomalous", edgecolor="#1a1d29"
            )
        axes[1, 0].set_title("Amount Distribution", color="#00d4ff")
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].set_xlabel("Amount ($)")

    # 4. Model agreement
    if "model_votes" in df.columns:
        vote_counts = df["model_votes"].value_counts().sort_index()
        axes[1, 1].bar(
            vote_counts.index, vote_counts.values,
            color=["#00d4ff", "#ffa500", "#ff6644", "#ff0000"][:len(vote_counts)],
            alpha=0.85,
        )
        axes[1, 1].set_title("Model Agreement Distribution", color="#00d4ff")
        axes[1, 1].set_xlabel("Models Flagging")
        axes[1, 1].set_ylabel("Count")

    filepath = os.path.join(output_dir, "system_overview.png")
    fig.savefig(filepath, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[Visualization] System overview saved to {filepath}")
    return filepath
