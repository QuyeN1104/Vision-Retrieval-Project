"""
src/search/reranker.py — Reranking functions to refine retrieval results.
Owner: Pipeline Engineer (PE-3, PE-4, Sprint 2/3)

  - Sprint 2 (PE-3): cosine_rerank — exact cosine re-scoring of FAISS candidates.
  - Sprint 3 (PE-2): mmr_rerank   — Maximal Marginal Relevance for result diversity.

``candidate_records`` are plain metadata dicts (the metadata.json schema:
{id, path, label, caption, split}); both ``id`` and ``image_id`` keys are accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class RetrievalResult:
    """Represents a structured query result."""
    image_id: str
    score: float
    path: str
    label: Optional[str] = None
    caption: Optional[str] = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.atleast_2d(np.asarray(mat, dtype=np.float32))
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def _to_result(record: dict, score: float) -> RetrievalResult:
    return RetrievalResult(
        image_id=str(record.get("image_id", record.get("id", ""))),
        score=float(score),
        path=record.get("path", ""),
        label=record.get("label"),
        caption=record.get("caption"),
    )


# --------------------------------------------------------------------------- #
# PE-3 (Sprint 2): exact cosine re-ranking
# --------------------------------------------------------------------------- #
def cosine_rerank(
    query_vec: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_records: List[dict],
) -> List[RetrievalResult]:
    """
    Perform exact cosine similarity re-scoring for retrieved candidates.

    Args:
        query_vec: 1D numpy array representing query embedding.
        candidate_embeddings: 2D numpy array of candidate embeddings (num_candidates, dim).
        candidate_records: List of dict metadata records of the candidates.

    Returns:
        List of RetrievalResult sorted in descending order of similarity score.
    """
    if len(candidate_records) == 0:
        return []
    q = _l2_normalize(query_vec)[0]
    cand = _l2_normalize(candidate_embeddings)
    sims = cand @ q  # cosine similarity (both sides unit-norm)
    order = np.argsort(-sims)
    return [_to_result(candidate_records[i], sims[i]) for i in order]


# --------------------------------------------------------------------------- #
# PE-2 (Sprint 3): MMR diversity re-ranking
# --------------------------------------------------------------------------- #
def mmr_rerank(
    query_vec: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_records: List[dict],
    k: int = 10,
    lambda_param: float = 0.5,
) -> List[RetrievalResult]:
    """
    Apply Maximal Marginal Relevance (MMR) to diversify retrieved results.

    MMR greedily selects the candidate that maximises
        lambda * sim(query, cand) - (1 - lambda) * max_{s in selected} sim(cand, s)
    balancing relevance to the query against redundancy with already-picked items.

    Args:
        query_vec: 1D numpy array representing query embedding.
        candidate_embeddings: 2D numpy array of candidate embeddings (num_candidates, dim).
        candidate_records: List of metadata dicts corresponding to candidates.
        k: Final number of diversified results to return.
        lambda_param: Diversity balance factor in [0, 1] (1: relevance only, 0: diversity only).

    Returns:
        Diversified List of RetrievalResult in MMR selection order; ``score`` is the
        candidate's cosine relevance to the query.
    """
    n = len(candidate_records)
    if n == 0:
        return []
    lam = float(np.clip(lambda_param, 0.0, 1.0))
    q = _l2_normalize(query_vec)[0]
    cand = _l2_normalize(candidate_embeddings)
    relevance = cand @ q                 # (n,)
    pairwise = cand @ cand.T             # (n, n) candidate-candidate similarity

    selected: List[int] = []
    remaining = list(range(n))
    k = min(int(k), n)
    while len(selected) < k:
        if not selected:
            best = int(remaining[int(np.argmax(relevance[remaining]))])
        else:
            best, best_score = remaining[0], -np.inf
            for i in remaining:
                diversity = float(np.max(pairwise[i, selected]))
                mmr = lam * float(relevance[i]) - (1.0 - lam) * diversity
                if mmr > best_score:
                    best_score, best = mmr, i
        selected.append(best)
        remaining.remove(best)

    return [_to_result(candidate_records[i], relevance[i]) for i in selected]
