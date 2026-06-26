"""
src/model/embeddings_cache.py — Embeddings and IDs caching layer.
Owner: Model Engineer (ME-1, ME-2, Sprint 2)
"""
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np


class EmbeddingsCache:
    """
    Handles saving and loading of precomputed embeddings and their associated IDs.
    """

    @staticmethod
    def save(embeddings: np.ndarray, ids: List[str], base_path: Path | str) -> None:
        """
        Save embeddings matrix as a .npy file and IDs list as a JSON mapping file.

        Args:
            embeddings: 2D numpy array of shape (num_items, embedding_dim).
            ids: List of unique identifiers (e.g., image paths or IDs).
            base_path: Base filepath (without extension) for saving cache files.
        """
        # Step 1: Validate — number of IDs must match number of embedding rows
        # If they don't match, the mapping between vector index → image ID is broken
        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings but {len(ids)} IDs. "
                f"Each embedding row must have exactly one corresponding ID."
            )

        base_path = Path(base_path)

        # Step 2: Create parent directories if they don't exist
        # e.g. base_path = "data/index/embeddings" → creates "data/index/"
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 3: Save the embedding matrix as .npy (numpy binary format)
        # .npy is fast and compact: 10,000 × 512 float32 = ~20MB on disk
        np.save(str(base_path) + ".npy", embeddings)

        # Step 4: Save the ID list as .json (human-readable mapping)
        # Index i in the .npy file corresponds to ids[i] in the .json file
        ids_path = str(base_path) + "_ids.json"
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(base_path: Path | str) -> Tuple[np.ndarray, List[str]]:
        """
        Load embeddings matrix and IDs list from the cache files.

        Args:
            base_path: Base filepath (without extension) of the cached files.

        Returns:
            A tuple of (embeddings numpy array, list of string IDs).
        """
        base_path = Path(base_path)
        npy_path = str(base_path) + ".npy"
        ids_path = str(base_path) + "_ids.json"

        # Step 1: Check that both cache files exist before attempting to load
        # Give a clear error message pointing to the exact missing file
        if not Path(npy_path).exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {npy_path}. "
                f"Run the indexing pipeline first to generate embeddings."
            )
        if not Path(ids_path).exists():
            raise FileNotFoundError(
                f"IDs mapping file not found: {ids_path}. "
                f"The cache may be corrupted — re-run indexing."
            )

        # Step 2: Load the numpy matrix and the JSON id list
        embeddings = np.load(npy_path).astype(np.float32)

        with open(ids_path, "r", encoding="utf-8") as f:
            ids = json.load(f)

        # Step 3: Validate consistency — same check as save(), guards against
        # manual edits or partial writes that corrupted the cache
        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"Cache corrupted: {embeddings.shape[0]} embeddings but {len(ids)} IDs in {ids_path}. "
                f"Delete both files and re-run indexing."
            )

        return embeddings, ids
