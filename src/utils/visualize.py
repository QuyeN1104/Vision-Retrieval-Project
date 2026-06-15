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
    raise NotImplementedError("To be implemented by UI Engineer (UI-4) in Sprint 1")
