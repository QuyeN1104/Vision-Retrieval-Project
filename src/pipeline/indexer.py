"""
src/pipeline/indexer.py — Orchestrator for building/generating index.
Owner: Tech Lead (TL-2, Sprint 2)
"""
from pathlib import Path


class Indexer:
    """
    Orchestrates the entire offline pipeline: Load dataset -> Batch encode -> Build Index -> Save.
    """

    def __init__(self, data_dir: Path | str, model_name: str, device: str = "cpu"):
        """
        Setup components for indexing.

        Args:
            data_dir: Source folder containing image database.
            model_name: Name of CLIP model to use.
            device: Computing device ('cpu', 'cuda').
        """
        raise NotImplementedError("To be implemented in Sprint 2")

    def run(self, output_dir: Path | str) -> None:
        """
        Execute indexing flow end-to-end and write files to the output directory.

        Args:
            output_dir: Target directory to save FAISS index, metadata, and cache.
        """
        raise NotImplementedError("To be implemented by Tech Lead (TL-2) in Sprint 2")
