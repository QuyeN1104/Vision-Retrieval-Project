"""
tests/unit/test_faiss_index.py — Unit tests for FaissIndex class.
Owner: Tech Lead (TL-3, Sprint 2)
"""
import tempfile
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.search.faiss_index import FaissIndex  # noqa: E402


def test_init_validation():
    # Valid initializations
    idx = FaissIndex(dim=128, index_type="flat")
    assert idx.dim == 128
    assert idx.index_type == "flat"

    idx_ivf = FaissIndex(dim=128, index_type="ivf", nlist=50, nprobe=5)
    assert idx_ivf.dim == 128
    assert idx_ivf.index_type == "ivf"
    assert idx_ivf.nlist == 50
    assert idx_ivf.nprobe == 5

    # Invalid initializations
    with pytest.raises(ValueError, match="dim must be positive"):
        FaissIndex(dim=0)
    with pytest.raises(ValueError, match="dim must be positive"):
        FaissIndex(dim=-5)
    with pytest.raises(ValueError, match="index_type must be one of flat/ivf"):
        FaissIndex(dim=128, index_type="invalid_type")


def test_build_and_search_flat():
    dim = 64
    rng = np.random.default_rng(42)
    embeddings = rng.normal(0, 1, (10, dim)).astype(np.float32)

    index = FaissIndex(dim=dim, index_type="flat")
    # Flat indices are trained by default upon creation in FAISS
    assert index.is_trained
    assert len(index) == 0

    index.build(embeddings)
    assert index.is_trained
    assert len(index) == 10

    # Search
    query = rng.normal(0, 1, (1, dim)).astype(np.float32)
    distances, indices = index.search(query, k=3)
    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert all(idx != -1 for idx in indices[0])


def test_build_and_search_ivf():
    dim = 64
    rng = np.random.default_rng(42)
    embeddings = rng.normal(0, 1, (150, dim)).astype(np.float32)

    index = FaissIndex(dim=dim, index_type="ivf", nlist=10)
    assert not index.is_trained
    assert len(index) == 0

    index.build(embeddings)
    assert index.is_trained
    assert len(index) == 150

    # Search
    query = rng.normal(0, 1, (dim,)).astype(np.float32)
    distances, indices = index.search(query, k=5)
    assert distances.shape == (1, 5)
    assert indices.shape == (1, 5)


def test_filter_search():
    dim = 8
    rng = np.random.default_rng(42)
    embeddings = rng.normal(0, 1, (10, dim)).astype(np.float32)
    
    index = FaissIndex(dim=dim, index_type="flat")
    index.build(embeddings)
    
    # 0, 2, 4 are 'cat'; 1, 3, 5 are 'dog'; others are None
    id_to_label = {
        0: "cat",
        2: "cat",
        4: "cat",
        1: "dog",
        3: "dog",
        5: "dog",
    }
    
    query = rng.normal(0, 1, (dim,)).astype(np.float32)
    
    # Search for cats
    distances, indices = index.filter_search(query, k=2, label_filter="cat", id_to_label=id_to_label)
    assert indices.shape[1] <= 2
    for idx in indices[0]:
        assert id_to_label.get(idx) == "cat"


def test_save_and_load():
    dim = 32
    rng = np.random.default_rng(42)
    embeddings = rng.normal(0, 1, (20, dim)).astype(np.float32)

    index = FaissIndex(dim=dim, index_type="flat")
    index.build(embeddings)

    query = rng.normal(0, 1, (dim,)).astype(np.float32)
    orig_dist, orig_idx = index.search(query, k=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.index"
        index.save(save_path)
        
        # Load index
        loaded_index = FaissIndex(dim=dim)
        loaded_index.load(save_path)
        
        assert len(loaded_index) == 20
        assert loaded_index.dim == dim
        
        loaded_dist, loaded_idx = loaded_index.search(query, k=5)
        np.testing.assert_array_almost_equal(orig_dist, loaded_dist)
        np.testing.assert_array_equal(orig_idx, loaded_idx)


def test_reconstruct():
    dim = 16
    rng = np.random.default_rng(42)
    embeddings = rng.normal(0, 1, (5, dim)).astype(np.float32)

    index = FaissIndex(dim=dim, index_type="flat")
    index.build(embeddings)

    reconstructed = index.reconstruct([0, 2])
    assert reconstructed.shape == (2, dim)
    
    # Check normalization alignment since FaissIndex normalizes L2
    import faiss
    expected = embeddings.copy()
    faiss.normalize_L2(expected)
    np.testing.assert_array_almost_equal(reconstructed[0], expected[0])
    np.testing.assert_array_almost_equal(reconstructed[1], expected[2])
