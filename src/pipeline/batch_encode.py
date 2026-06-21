"""
src/pipeline/batch_encode.py — Batch encoding helper.
Owner: Pipeline Engineer (PE-4, Sprint 1)

Encodes an entire image dataset into a normalized embeddings matrix, streaming
the work in chunks so memory stays bounded for large datasets. Row ``i`` of the
returned matrix corresponds to ``records[i]`` (alignment is preserved so the
caller can build a position-aligned FAISS index + metadata).

The ``ImageRecord`` / ``CLIPEncoder`` types are owned by the Data / Model
Engineers; we depend only on their documented interfaces:
  * ``record.path``  -> image file path (also tolerates a dict ``record["path"]``)
  * ``encoder.encode_images_batch(images, batch_size) -> np.ndarray (n, dim)``
They are imported under ``TYPE_CHECKING`` only, so this module stays import-safe
before those files land.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # avoid a hard runtime dependency on teammates' unfinished modules
    from src.data.dataset_loader import ImageRecord
    from src.model.clip_encoder import CLIPEncoder

logger = logging.getLogger(__name__)


def _record_path(record) -> str:
    """Get the image path from an ImageRecord (object) or a dict record."""
    if hasattr(record, "path"):
        return record.path
    if isinstance(record, dict) and "path" in record:
        return record["path"]
    raise AttributeError(f"Record has no 'path': {record!r}")


def encode_dataset(
    records: "List[ImageRecord]",
    encoder: "CLIPEncoder",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Load images iteratively, process them, and run batch inference to extract embeddings.

    Args:
        records: List of ImageRecords to encode (must expose ``.path``).
        encoder: CLIPEncoder wrapper instance.
        batch_size: Number of images per forward pass batch.

    Returns:
        2D numpy array of shape (num_records, embedding_dim) containing L2-normalized
        embeddings, aligned row-for-row with ``records``.

    Raises:
        ValueError: if ``records`` is empty.
        RuntimeError: if an image fails to load (the dataset must be pre-validated by
            the Data Engineer's preprocessor; skipping would desync row alignment).
    """
    if not records:
        raise ValueError("encode_dataset received an empty record list.")
    batch_size = max(1, int(batch_size))

    chunks: List[np.ndarray] = []
    total = len(records)
    for start in range(0, total, batch_size):
        batch = records[start:start + batch_size]
        images = []
        for rec in batch:
            path = _record_path(rec)
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as e:  # fail fast: a skipped image would misalign the index
                raise RuntimeError(
                    f"Failed to load image {path!r}: {e}. Clean the dataset "
                    f"(Data Engineer preprocessor) before encoding so row alignment holds."
                ) from e

        emb = np.asarray(encoder.encode_images_batch(images, batch_size=batch_size),
                         dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != len(images):
            raise RuntimeError(
                f"Encoder returned shape {emb.shape} for {len(images)} images "
                f"(expected (n, dim))."
            )
        chunks.append(emb)
        logger.info("Encoded %d/%d images.", min(start + batch_size, total), total)

    embeddings = np.vstack(chunks).astype(np.float32)
    assert embeddings.shape[0] == total, "embedding/record count mismatch"
    logger.info("Dataset encoded: %d embeddings, dim=%d.", *embeddings.shape)
    return embeddings
