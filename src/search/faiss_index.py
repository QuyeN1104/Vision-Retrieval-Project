"""
src/search/faiss_index.py — FAISS index management and querying.
Owner: Pipeline Engineer (PE-1, PE-2, PE-3, PE-4, Sprint 1/2/3)
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np


class FaissIndex:
    """
    Manages building, saving, loading, and searching in a FAISS index.
    """

    def __init__(self, dim: int = 512, index_type: str = "flat", **kwargs):
        """
        Initialize the FAISS index settings.

        Args:
            dim: Dimension of embeddings (default: 512).
            index_type: Type of index to use ("flat" for FlatIP, "ivf" for IVFFlat).
            kwargs: Extra parameters like nlist for IVF.
        """
        raise NotImplementedError("To be implemented by Pipeline Engineer (PE-1) in Sprint 1")

    def build(self, embeddings: np.ndarray) -> None:
        """
        Train and populate the FAISS index with dataset embeddings.

        Args:
            embeddings: 2D numpy array of shape (num_items, dim).
        """
        raise NotImplementedError("To be implemented by Pipeline Engineer (PE-2) in Sprint 1")

    def save(self, path: Path | str) -> None:
        """
        Serialize the FAISS index to a file on disk.

        Args:
            path: Target file path to write.
        """
        raise NotImplementedError("To be implemented by Pipeline Engineer (PE-3) in Sprint 1")

    def load(self, path: Path | str) -> None:
        """
        Load a serialized FAISS index from disk.

        Args:
            path: Source file path to read.
        """
        raise NotImplementedError("To be implemented by Pipeline Engineer (PE-4) in Sprint 1")

    def search(self, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Approximate Nearest Neighbor (ANN) search.

        Args:
            query_vec: 1D numpy array of shape (dim,) or 2D array of shape (1, dim).
            k: Number of nearest neighbors to retrieve.

        Returns:
            A tuple of (distances numpy array, indices numpy array).
        """
        raise NotImplementedError("To be implemented by Pipeline Engineer (PE-1) in Sprint 2")

    def filter_search(self, query_vec: np.ndarray, k: int, label_filter: str, id_to_label: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform search while applying a category filter.

        Args:
            query_vec: 1D numpy array of shape (dim,).
            k: Number of results to retrieve.
            label_filter: The category label to filter by.
            id_to_label: Mapping from index ID to its corresponding class label.

        Returns:
            A tuple of (distances numpy array, indices numpy array).
        """
        raise NotImplementedError("To be implemented by Pipeline Engineer (PE-2) in Sprint 2")
