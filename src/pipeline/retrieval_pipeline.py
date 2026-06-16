"""
src/pipeline/retrieval_pipeline.py — End-to-end query orchestration pipeline.
Owner: Pipeline Engineer (PE-4, PE-1, Sprint 2/3)

Flow:  text/image -> CLIP encode -> FAISS search (over-fetch) -> cosine rerank
       -> attach metadata -> top-K RetrievalResult.

The production constructor wires the real CLIPEncoder (Model Engineer) + a saved
FAISS index + metadata.json. ``from_components`` injects pre-built parts, which
keeps the pipeline unit/integration testable before the encoder is implemented.

Metadata is a position-aligned list of records (metadata.json schema
{id, path, label, caption, split}); list position == FAISS index id, because the
index was built from the same record order (see batch_encode.encode_dataset).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from PIL import Image

from src.search.faiss_index import FaissIndex
from src.search.reranker import RetrievalResult, cosine_rerank

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    Main pipeline orchestrating the query flow: encode inputs -> search index -> rerank.
    """

    def __init__(self, encoder_model: str, index_path: Path | str, metadata_path: Path | str,
                 dim: int = 512, rerank: bool = True, overfetch: int = 4):
        """
        Load assets required for real-time querying.

        Args:
            encoder_model: Model name/path for the CLIP encoder.
            index_path: Path to the serialized FAISS index.
            metadata_path: Path to the serialized metadata file (metadata.json).
            dim: Embedding dimension (default 512).
            rerank: Whether to apply exact cosine re-ranking after ANN search.
            overfetch: Candidate multiplier fetched before re-ranking (k * overfetch).
        """
        from src.model.clip_encoder import CLIPEncoder  # imported lazily (Model Engineer's)

        self.encoder = CLIPEncoder(encoder_model)
        self.index = FaissIndex(dim=dim)
        self.index.load(index_path)
        self.metadata = self._load_metadata(metadata_path)
        self.rerank = rerank
        self.overfetch = max(1, int(overfetch))
        logger.info("RetrievalPipeline ready: %d items indexed.", self.index.ntotal)

    # --------------------------------------------------------- DI / testing ctor
    @classmethod
    def from_components(cls, encoder, index: FaissIndex, metadata: List[dict],
                        rerank: bool = True, overfetch: int = 4) -> "RetrievalPipeline":
        """Build a pipeline from already-constructed parts (for tests / custom wiring)."""
        obj = cls.__new__(cls)
        obj.encoder = encoder
        obj.index = index
        obj.metadata = list(metadata)
        obj.rerank = rerank
        obj.overfetch = max(1, int(overfetch))
        return obj

    # ------------------------------------------------------------------ loaders
    @staticmethod
    def _load_metadata(metadata_path: Path | str) -> List[dict]:
        data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        if not isinstance(data, list):
            raise ValueError(f"metadata must be a list of records, got {type(data)}")
        return data

    # --------------------------------------------------------- PE-4 (Sprint 2)
    def query_by_text(self, text: str, k: int = 10,
                      label_filter: Optional[str] = None) -> List[RetrievalResult]:
        """
        Search for images matching a text prompt.

        Args:
            text: Input text query.
            k: Number of nearest neighbors to retrieve.
            label_filter: Optional class label to filter candidates.

        Returns:
            List of RetrievalResult sorted by relevance.
        """
        query_vec = self.encoder.encode_text(text)
        return self._retrieve(query_vec, k, label_filter)

    # --------------------------------------------------------- PE-1 (Sprint 3)
    def query_by_image(self, image: Image.Image, k: int = 10,
                       label_filter: Optional[str] = None) -> List[RetrievalResult]:
        """
        Search for images matching an uploaded query image (Reverse Image Search).

        Args:
            image: Uploaded PIL Image.
            k: Number of nearest neighbors to retrieve.
            label_filter: Optional class label to filter candidates.

        Returns:
            List of RetrievalResult sorted by relevance.
        """
        query_vec = self.encoder.encode_image(image)
        return self._retrieve(query_vec, k, label_filter)

    # ------------------------------------------------------------ shared core
    def _retrieve(self, query_vec, k: int, label_filter: Optional[str]) -> List[RetrievalResult]:
        if self.index.ntotal == 0:
            return []
        fetch = k * self.overfetch if self.rerank else k

        if label_filter is not None:
            id_to_label = {i: rec.get("label") for i, rec in enumerate(self.metadata)}
            distances, indices = self.index.filter_search(query_vec, fetch, label_filter, id_to_label)
        else:
            distances, indices = self.index.search(query_vec, fetch)

        ids = [int(i) for i in indices[0] if int(i) != -1]
        if not ids:
            return []

        if self.rerank:
            cand_emb = self.index.reconstruct(ids)
            cand_recs = [self._record(i) for i in ids]
            return cosine_rerank(query_vec, cand_emb, cand_recs)[:k]

        # no rerank: use FAISS scores directly
        from src.search.reranker import _to_result
        return [_to_result(self._record(i), float(d))
                for d, i in zip(distances[0], ids)][:k]

    def _record(self, idx: int) -> dict:
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return {"id": str(idx), "path": "", "label": None, "caption": None}
