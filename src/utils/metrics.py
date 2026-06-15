"""
src/utils/metrics.py — Retrieval evaluation metrics.
Owner: Data Engineer (DE-2, DE-3, Sprint 3) + Model Engineer (ME-2, Sprint 3)
"""
from typing import List, Set


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute Recall@K: fraction of relevant items found in top-K results.

    Args:
        retrieved: Ordered list of retrieved image IDs.
        relevant: Set of ground-truth relevant image IDs.
        k: Cutoff rank.

    Returns:
        Recall@K score in [0.0, 1.0].
    """
    raise NotImplementedError("To be implemented by Data Engineer (DE-2) in Sprint 3")


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Compute Average Precision (AP) for a single query.

    Args:
        retrieved: Ordered list of retrieved image IDs.
        relevant: Set of ground-truth relevant image IDs.

    Returns:
        AP score in [0.0, 1.0].
    """
    raise NotImplementedError("To be implemented by Data Engineer (DE-3) in Sprint 3")


def mean_ap(ap_scores: List[float]) -> float:
    """
    Mean Average Precision (mAP) over all queries.

    Args:
        ap_scores: List of AP scores for each test query.

    Returns:
        mAP score in [0.0, 1.0].
    """
    raise NotImplementedError("To be implemented by Data Engineer (DE-3) in Sprint 3")


def mrr_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute Mean Reciprocal Rank at K for a single query.

    Args:
        retrieved: Ordered list of retrieved image IDs.
        relevant: Set of ground-truth relevant image IDs.
        k: Cutoff rank.

    Returns:
        1/rank of the first relevant result, or 0.0 if none found in top-K.
    """
    raise NotImplementedError("To be implemented by Model Engineer (ME-2) in Sprint 3")
