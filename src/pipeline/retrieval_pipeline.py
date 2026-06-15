"""
src/pipeline/retrieval_pipeline.py — End-to-end query orchestration pipeline.
Owner: Pipeline Engineer (PE-4, PE-1, Sprint 2/3)
"""
from pathlib import Path
from PIL import Image
from typing import List, Optional
from src.search.reranker import RetrievalResult


class RetrievalPipeline:
    """
    Main pipeline orchestrating the query flow: encode inputs -> search index -> rerank.
    """

    def __init__(self, encoder_model: str, index_path: Path | str, metadata_path: Path | str):
        """
        Load assets required for real-time querying.

        Args:
            encoder_model: Model name/path for CLIP encoder.
            index_path: Path to serialized FAISS index.
            metadata_path: Path to serialized metadata file.
        """
        raise NotImplementedError("To be implemented in Sprint 2")

    def query_by_text(self, text: str, k: int = 10, label_filter: Optional[str] = None) -> List[RetrievalResult]:
        """
        Search for images matching a text prompt.

        Args:
            text: Input text query.
            k: Number of nearest neighbors to retrieve.
            label_filter: Optional class label to filter candidates.

        Returns:
            List of RetrievalResult sorted by relevance.
        """
        raise NotImplementedError("To be implemented by Pipeline Engineer (PE-4) in Sprint 2")

    def query_by_image(self, image: Image.Image, k: int = 10, label_filter: Optional[str] = None) -> List[RetrievalResult]:
        """
        Search for images matching an uploaded query image (Reverse Image Search).

        Args:
            image: Uploaded PIL Image.
            k: Number of nearest neighbors to retrieve.
            label_filter: Optional class label to filter candidates.

        Returns:
            List of RetrievalResult sorted by relevance.
        """
        raise NotImplementedError("To be implemented by Pipeline Engineer (PE-1) in Sprint 3")
