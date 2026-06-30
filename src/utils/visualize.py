"""
src/utils/visualize.py — Visualization helpers.
Owner: UI Engineer (UI-4, Sprint 1)
"""
from typing import List

import matplotlib.pyplot as plt


def draw_score_bar(scores: List[float], image_ids: List[str], top_k: int = 10) -> plt.Figure:
    """
    Draw a horizontal bar chart of similarity scores.

    Args:
        scores: List of cosine similarity scores.
        image_ids: Corresponding image IDs/filenames.
        top_k: Number of results to display.

    Returns:
        Matplotlib Figure object (can be passed to st.pyplot).
    """
    # Limit to top_k entries
    display_scores = scores[:top_k]
    display_ids = image_ids[:top_k]

    # Reverse so highest score appears at the top of the chart
    display_scores = display_scores[::-1]
    display_ids = display_ids[::-1]

    fig, ax = plt.subplots(figsize=(10, max(3, len(display_ids) * 0.4)))

    colors = plt.cm.viridis([s for s in display_scores])
    bars = ax.barh(display_ids, display_scores, color=colors, edgecolor="white", height=0.6)

    ax.set_xlabel("Cosine Similarity")
    ax.set_title("Top-K Similarity Scores")
    ax.set_xlim(0, 1.05)

    # Add score labels on bars
    for bar, score in zip(bars, display_scores):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    return fig
