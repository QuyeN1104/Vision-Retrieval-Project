"""
src/search/faiss_index.py — FAISS index management and querying.
Owner: Pipeline Engineer (PE-1, PE-2, PE-3, PE-4, Sprint 1/2/3)

Implements:
  - Sprint 1: __init__ (FlatIP / IVFFlat), build (train + add), save / load.
  - Sprint 2: search (ANN), filter_search (metadata-filtered search).
  - Sprint 3: IVFFlat experiment knobs (nlist / nprobe) + reconstruct helper.

Embeddings are L2-normalised on the way in, so the inner-product index returns
cosine similarity directly. CLIP embeddings are already unit-norm, so this is a
safe no-op for them and guarantees cosine even if a caller forgets to normalise.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# index_type aliases accepted by the constructor
_FLAT = {"flat", "flatip", "flat_ip"}
_IVF = {"ivf", "ivfflat", "ivf_flat"}


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
            kwargs: Extra parameters — ``nlist`` (IVF cluster count, default 100) and
                ``nprobe`` (IVF clusters probed at search time, default 10).
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        norm_type = index_type.lower().replace("-", "_")
        if norm_type not in _FLAT | _IVF:
            raise ValueError(f"index_type must be one of flat/ivf, got {index_type!r}")

        self.dim = int(dim)
        self.index_type = "ivf" if norm_type in _IVF else "flat"
        self.nlist = int(kwargs.get("nlist", 100))
        self.nprobe = int(kwargs.get("nprobe", 10))
        self.index = self._new_index(self.nlist)
        logger.info("FaissIndex initialised: dim=%d type=%s nlist=%s",
                    self.dim, self.index_type, self.nlist if self.index_type == "ivf" else "-")

    # ------------------------------------------------------------------ helpers
    def _new_index(self, nlist: int):
        import faiss
        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.dim)
        quantizer = faiss.IndexFlatIP(self.dim)
        idx = faiss.IndexIVFFlat(quantizer, self.dim, max(1, nlist), faiss.METRIC_INNER_PRODUCT)
        idx.nprobe = self.nprobe
        return idx

    def _prepare(self, vecs: np.ndarray) -> np.ndarray:
        """Validate -> float32, C-contiguous, L2-normalised (so IP == cosine)."""
        import faiss
        arr = np.ascontiguousarray(np.atleast_2d(np.asarray(vecs)), dtype=np.float32)
        if arr.shape[1] != self.dim:
            raise ValueError(f"embedding dim {arr.shape[1]} != index dim {self.dim}")
        faiss.normalize_L2(arr)
        return arr

    @property
    def is_trained(self) -> bool:
        return bool(self.index.is_trained)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def __len__(self) -> int:
        return self.ntotal

    # --------------------------------------------------------------- PE-2 build
    def build(self, embeddings: np.ndarray) -> None:
        """
        Train and populate the FAISS index with dataset embeddings.

        Args:
            embeddings: 2D numpy array of shape (num_items, dim).
        """
        emb = self._prepare(embeddings)
        n = emb.shape[0]
        if n == 0:
            raise ValueError("Cannot build an index from 0 embeddings.")

        if self.index_type == "ivf":
            # IVF needs at least `nlist` training points; clamp for small corpora.
            effective_nlist = max(1, min(self.nlist, n))
            if effective_nlist != self.nlist:
                logger.warning("IVF nlist %d > n=%d; clamping to %d.",
                               self.nlist, n, effective_nlist)
                self.nlist = effective_nlist
                self.index = self._new_index(self.nlist)
            if not self.index.is_trained:
                self.index.train(emb)

        self.index.add(emb)
        if self.index_type == "ivf":
            self.index.make_direct_map()  # enable reconstruct() for the reranker
        logger.info("Index built: %d vectors (type=%s).", self.ntotal, self.index_type)

    # --------------------------------------------------------- PE-3 save / load
    def save(self, path: Path | str) -> None:
        """
        Serialize the FAISS index to a file on disk (plus a sidecar meta json).

        Args:
            path: Target file path to write.
        """
        import faiss
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        meta = {"dim": self.dim, "index_type": self.index_type,
                "nlist": self.nlist, "nprobe": self.nprobe, "ntotal": self.ntotal}
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Index saved to %s (%d vectors).", path, self.ntotal)

    def load(self, path: Path | str) -> None:
        """
        Load a serialized FAISS index from disk (restoring config from the sidecar).

        Args:
            path: Source file path to read.
        """
        import faiss
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        self.index = faiss.read_index(str(path))
        self.dim = int(self.index.d)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.index_type = meta.get("index_type", self.index_type)
            self.nlist = int(meta.get("nlist", self.nlist))
            self.nprobe = int(meta.get("nprobe", self.nprobe))
        if self.index_type == "ivf":
            self.index.nprobe = self.nprobe
            try:
                self.index.make_direct_map()
            except Exception:  # pragma: no cover - already has a direct map
                pass
        logger.info("Index loaded from %s (%d vectors, type=%s).",
                    path, self.ntotal, self.index_type)

    # ------------------------------------------------------------- PE-1 search
    def search(self, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Approximate Nearest Neighbor (ANN) search.

        Args:
            query_vec: 1D numpy array of shape (dim,) or 2D array of shape (1, dim).
            k: Number of nearest neighbors to retrieve.

        Returns:
            A tuple of (distances numpy array, indices numpy array), each shape (1, k).
        """
        if self.ntotal == 0:
            raise RuntimeError("Index is empty — call build() or load() first.")
        q = self._prepare(query_vec)
        k = max(1, min(int(k), self.ntotal))
        distances, indices = self.index.search(q, k)
        return distances, indices

    # ------------------------------------------------------- PE-2 filter_search
    def filter_search(self, query_vec: np.ndarray, k: int, label_filter: str,
                      id_to_label: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform search while applying a category filter.

        Over-fetches a wider candidate set, then keeps only candidates whose label
        matches ``label_filter`` and returns the top-``k`` of those.

        Args:
            query_vec: 1D numpy array of shape (dim,).
            k: Number of results to retrieve.
            label_filter: The category label to filter by.
            id_to_label: Mapping from index ID to its corresponding class label.

        Returns:
            A tuple of (distances, indices) numpy arrays, each shape (1, n<=k).
        """
        # widen the net so enough candidates survive the filter
        over_k = min(self.ntotal, max(int(k) * 10, 100))
        distances, indices = self.search(query_vec, over_k)
        keep_d, keep_i = [], []
        for d, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if id_to_label.get(int(idx)) == label_filter:
                keep_d.append(float(d))
                keep_i.append(int(idx))
                if len(keep_i) >= k:
                    break
        return (np.array([keep_d], dtype=np.float32), np.array([keep_i], dtype=np.int64))

    # ---------------------------------------------------- reconstruct (rerank)
    def reconstruct(self, indices) -> np.ndarray:
        """
        Return the stored (normalised) embeddings for the given index IDs.

        Used by the reranker to re-score candidates exactly. Works for FlatIP and,
        after ``build()`` (which calls ``make_direct_map``), for IVFFlat too.

        Args:
            indices: iterable of integer index IDs.

        Returns:
            2D numpy array of shape (len(indices), dim).
        """
        ids = [int(i) for i in indices]
        if not ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.index.reconstruct(i) for i in ids]).astype(np.float32)
