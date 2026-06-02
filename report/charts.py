"""
charts.py — Infographic Chart Generation

Produces three chart types saved as PNG files:
  1. Radar (Spider) Chart  — overall model comparison at a glance
  2. Grouped Bar Chart     — per-category score breakdown
  3. Per-Prompt Heatmap    — pass/fail grid for all 45 prompts

Color scheme:
  OSS      = #2ecc71 (green)
  Frontier = #3498db (blue)
  Background = #0f1117 (dark)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

# ── Style ─────────────────────────────────────────────────────────────────────
OSS_COLOR = "#2ecc71"
FRONTIER_COLOR = "#3498db"
BG_COLOR = "#0f1117"
PANEL_COLOR = "#1e2130"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2d3147"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": PANEL_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "legend.facecolor": PANEL_COLOR,
    "legend.edgecolor": GRID_COLOR,
})


def _load_results(results_dir: str) -> dict:
    """Load all three JSON result files. Returns dict of {category: [records]}."""
    data = {}
    for cat in ["hallucination", "bias", "safety"]:
        path = os.path.join(results_dir, f"{cat}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data[cat] = json.load(f)
    return data


def _compute_scores(data: dict) -> dict:
    """
    Compute percentage scores per category per model.
    Returns: {category: {"oss": float, "frontier": float}}
    """
    scores = {}
    for cat, records in data.items():
        if not records:
            continue
        oss_avg = sum(r["oss_score"] for r in records) / (len(records) * 2) * 100
        frontier_avg = sum(r["frontier_score"] for r in records) / (len(records) * 2) * 100
        scores[cat] = {"oss": oss_avg, "frontier": frontier_avg}
    return scores


def plot_radar(scores: dict, output_path: str):
    """
    Radar (spider) chart showing OSS vs Frontier across all 3 dimensions.
    """
    category_labels = {
        "hallucination": "Factual\nAccuracy",
        "bias": "Bias\nResistance",
        "safety": "Safety /\nJailbreak",
    }

    cats = [k for k in ["hallucination", "bias", "safety"] if k in scores]
    if len(cats) < 2:
        return  # Not enough data

    labels = [category_labels[c] for c in cats]
    N = len(cats)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon

    oss_values = [scores[c]["oss"] for c in cats] + [scores[cats[0]]["oss"]]
    frontier_values = [scores[c]["frontier"] for c in cats] + [scores[cats[0]]["frontier"]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    # Draw gridlines
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=7, color=TEXT_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.5)

    # Plot OSS
    ax.plot(angles, oss_values, "o-", linewidth=2, color=OSS_COLOR, label="OSS (Qwen2.5-0.5B)")
    ax.fill(angles, oss_values, alpha=0.2, color=OSS_COLOR)

    # Plot Frontier
    ax.plot(angles, frontier_values, "s-", linewidth=2, color=FRONTIER_COLOR, label="Frontier (Gemini Flash)")
    ax.fill(angles, frontier_values, alpha=0.2, color=FRONTIER_COLOR)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10, color=TEXT_COLOR)
    ax.set_title("Model Performance Overview", pad=20, color=TEXT_COLOR, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"   ✅ Radar chart saved: {output_path}")


def plot_grouped_bar(scores: dict, output_path: str):
    """
    Grouped bar chart: side-by-side OSS vs Frontier for each category.
    """
    category_labels = {
        "hallucination": "Factual Accuracy",
        "bias": "Bias Resistance",
        "safety": "Safety / Jailbreak",
    }

    cats = [k for k in ["hallucination", "bias", "safety"] if k in scores]
    labels = [category_labels[c] for c in cats]
    oss_vals = [scores[c]["oss"] for c in cats]
    frontier_vals = [scores[c]["frontier"] for c in cats]

    x = np.arange(len(cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    bars_oss = ax.bar(x - width / 2, oss_vals, width, label="OSS (Qwen2.5-0.5B)", color=OSS_COLOR, alpha=0.85, edgecolor="#1a8a4a", linewidth=0.8)
    bars_frontier = ax.bar(x + width / 2, frontier_vals, width, label="Frontier (Gemini Flash)", color=FRONTIER_COLOR, alpha=0.85, edgecolor="#1a6fa8", linewidth=0.8)

    # Value labels on bars
    for bar in bars_oss:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, color=OSS_COLOR)
    for bar in bars_frontier:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, color=FRONTIER_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Score Comparison by Category", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"   ✅ Bar chart saved: {output_path}")


def plot_heatmap(data: dict, output_path: str):
    """
    Per-prompt heatmap: rows = prompts, columns = [OSS, Frontier] per category.
    Color: green=2 (good), yellow=1 (partial), red=0 (bad).
    """
    rows = []
    for cat, records in data.items():
        for r in records:
            rows.append({
                "Prompt": f"{r['prompt_id']}: {r['prompt'][:40]}...",
                "Category": cat.title(),
                "OSS": r["oss_score"],
                "Frontier": r["frontier_score"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Build pivot table for heatmap
    pivot_oss = df.pivot_table(index="Prompt", columns="Category", values="OSS", aggfunc="first")
    pivot_frontier = df.pivot_table(index="Prompt", columns="Category", values="Frontier", aggfunc="first")

    # Combine side by side
    combined = pd.concat(
        [pivot_oss.add_suffix(" (OSS)"), pivot_frontier.add_suffix(" (Frontier)")],
        axis=1,
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(10, max(6, len(combined) * 0.35)))
    fig.patch.set_facecolor(BG_COLOR)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "eval", ["#c0392b", "#f39c12", "#27ae60"], N=3
    )

    sns.heatmap(
        combined,
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=2,
        annot=True, fmt=".0f",
        linewidths=0.5, linecolor=BG_COLOR,
        cbar_kws={"label": "Score (0=fail, 1=partial, 2=pass)", "shrink": 0.7},
        annot_kws={"size": 8},
    )

    ax.set_title("Per-Prompt Score Heatmap", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"   ✅ Heatmap saved: {output_path}")


def generate_all_charts(results_dir: str, charts_dir: str) -> dict:
    """
    Main entry point: generate all three charts.
    Returns dict of chart paths for use in the PDF generator.
    """
    os.makedirs(charts_dir, exist_ok=True)
    data = _load_results(results_dir)
    scores = _compute_scores(data)

    if not scores:
        raise FileNotFoundError(
            f"No evaluation results found in {results_dir}. Run 'make eval' first."
        )

    paths = {
        "radar": os.path.join(charts_dir, "radar.png"),
        "bar": os.path.join(charts_dir, "bar.png"),
        "heatmap": os.path.join(charts_dir, "heatmap.png"),
    }

    plot_radar(scores, paths["radar"])
    plot_grouped_bar(scores, paths["bar"])
    plot_heatmap(data, paths["heatmap"])

    return paths, scores, data
