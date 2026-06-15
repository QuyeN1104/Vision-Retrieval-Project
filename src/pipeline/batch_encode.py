"""
src/pipeline/batch_encode.py — Batch encoding helper.
Owner: Pipeline Engineer (PE-4, Sprint 1)
"""
from typing import List
import numpy as np
from src.data.dataset_loader import ImageRecord
from src.model.clip_encoder import CLIPEncoder


def encode_dataset(
    records: List[ImageRecord],
    encoder: CLIPEncoder,
    batch_size: int = 32
) -> np.ndarray:
    """
    Load images iteratively, process them, and run batch inference to extract embeddings.

    Args:
        records: List of ImageRecords to encode.
        encoder: CLIPEncoder wrapper instance.
        batch_size: Number of images per forward pass batch.

    Returns:
        2D numpy array of shape (num_records, embedding_dim) containing normalized embeddings.
    """
    raise NotImplementedError("To be implemented by Pipeline Engineer (PE-4) in Sprint 1")
