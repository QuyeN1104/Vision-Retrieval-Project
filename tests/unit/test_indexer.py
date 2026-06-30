"""
tests/unit/test_indexer.py — Unit tests for the Indexer class.
Owner: Tech Lead (TL-2, Sprint 2)
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.datatypes import ImageRecord  # noqa: E402
from src.pipeline.indexer import Indexer  # noqa: E402


@patch("src.pipeline.indexer.CLIPEncoder")
@patch("src.pipeline.indexer.load_dataset")
@patch("src.pipeline.indexer.save_metadata")
@patch("src.pipeline.indexer.FaissIndex")
@patch("src.pipeline.batch_encode.encode_dataset")
def test_indexer_run_success(
    mock_encode_dataset,
    mock_faiss_index_cls,
    mock_save_metadata,
    mock_load_dataset,
    mock_clip_encoder_cls,
):
    # Set up mocks
    mock_encoder = MagicMock()
    # Mock dynamic projection_dim lookup
    mock_encoder.model.config.projection_dim = 512
    mock_clip_encoder_cls.return_value = mock_encoder

    # Mock dataset load
    records = [
        ImageRecord(id="1", path="data/images/car_1.jpg"),
        ImageRecord(id="2", path="data/images/cat_1.jpg"),
    ]
    mock_load_dataset.return_value = records

    # Mock batch encode output
    embeddings = np.random.randn(2, 512).astype(np.float32)
    mock_encode_dataset.return_value = embeddings

    # Mock FaissIndex
    mock_faiss_index = MagicMock()
    mock_faiss_index_cls.return_value = mock_faiss_index

    # Initialize Indexer
    indexer = Indexer(data_dir="dummy_data", model_name="dummy_model", device="cpu")

    # Run indexer
    with patch("pathlib.Path.mkdir"):
        indexer.run(output_dir="dummy_output")

    # Assertions
    mock_load_dataset.assert_called_once_with("dummy_data")
    mock_encode_dataset.assert_called_once_with(records, mock_encoder, batch_size=32)
    mock_faiss_index_cls.assert_called_once_with(dim=512, index_type="flat")
    mock_faiss_index.build.assert_called_once_with(embeddings)
    mock_faiss_index.save.assert_called_once()
    mock_save_metadata.assert_called_once_with(records, "dummy_output/metadata.json")


@patch("src.pipeline.indexer.CLIPEncoder")
@patch("src.pipeline.indexer.load_dataset")
def test_indexer_run_empty_dataset(mock_load_dataset, mock_clip_encoder_cls):
    mock_clip_encoder_cls.return_value = MagicMock()
    mock_load_dataset.return_value = []

    indexer = Indexer(data_dir="dummy_data", model_name="dummy_model", device="cpu")

    with pytest.raises(ValueError, match="No valid images"):
        indexer.run(output_dir="dummy_output")
