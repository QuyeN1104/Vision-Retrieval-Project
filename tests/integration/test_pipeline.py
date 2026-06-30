"""
tests/integration/test_pipeline.py — Retrieval pipeline integration tests.
Owner: Pipeline Engineer (PE-4, Sprint 3)

Verifies the end-to-end Pipeline-Engineer stack (FaissIndex + reranker +
RetrievalPipeline) without depending on the real CLIP encoder: a deterministic
FakeEncoder injects query vectors via RetrievalPipeline.from_components.

Core acceptance (PE-4): a text query returns top-5 results with the correct label.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# make the repo root importable (so `import src...` works) when run from anywhere
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.retrieval_pipeline import RetrievalPipeline  # noqa: E402
from src.search.faiss_index import FaissIndex  # noqa: E402
from src.search.reranker import cosine_rerank, mmr_rerank  # noqa: E402

DIM = 16


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeEncoder:
    """Returns a preset vector per text key / passes image vectors straight through."""

    def __init__(self, text_vecs: dict):
        self.text_vecs = text_vecs

    def encode_text(self, text):
        return self.text_vecs[text]

    def encode_image(self, image):           # the test passes a vector as the "image"
        return np.asarray(image, dtype=np.float32)


@pytest.fixture()
def pipeline():
    """Two well-separated clusters ('cat' near e0, 'dog' near e1), 20 images each."""
    rng = np.random.default_rng(0)
    c_cat = np.zeros(DIM, np.float32)
    c_cat[0] = 1.0
    c_dog = np.zeros(DIM, np.float32)
    c_dog[1] = 1.0

    embs, metadata = [], []
    for label, center in (("cat", c_cat), ("dog", c_dog)):
        for j in range(20):
            vec = _unit(center + rng.normal(0, 0.05, DIM).astype(np.float32))
            embs.append(vec)
            metadata.append({"id": f"{label}_{j}", "path": f"/img/{label}_{j}.jpg", "label": label})
    embeddings = np.vstack(embs).astype(np.float32)

    index = FaissIndex(dim=DIM, index_type="flat")
    index.build(embeddings)
    encoder = FakeEncoder({"cat": c_cat, "dog": c_dog})
    return RetrievalPipeline.from_components(encoder, index, metadata), c_cat, c_dog


def test_text_query_returns_correct_label(pipeline):
    """PE-4 acceptance: top-5 results for a 'cat' query are all cats."""
    pipe, _c_cat, _c_dog = pipeline
    results = pipe.query_by_text("cat", k=5)
    assert len(results) == 5
    assert all(r.label == "cat" for r in results)
    # scores are descending cosine similarities in [0, 1]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].score > 0.9


def test_query_by_image(pipeline):
    pipe, _c_cat, c_dog = pipeline
    results = pipe.query_by_image(c_dog, k=5)
    assert len(results) == 5 and all(r.label == "dog" for r in results)


def test_label_filter_overrides_query(pipeline):
    """A 'cat' query filtered to 'dog' must return only dog images."""
    pipe, _c_cat, _c_dog = pipeline
    results = pipe.query_by_text("cat", k=5, label_filter="dog")
    assert len(results) >= 1
    assert all(r.label == "dog" for r in results)


def test_result_fields_populated(pipeline):
    pipe, _c_cat, _c_dog = pipeline
    r = pipe.query_by_text("cat", k=1)[0]
    assert r.image_id and r.path and r.label == "cat" and isinstance(r.score, float)


def test_cosine_rerank_orders_by_similarity():
    q = _unit(np.array([1, 0, 0, 0], np.float32))
    cand = np.array([[0.2, 1, 0, 0], [0.95, 0.1, 0, 0], [0.6, 0.6, 0, 0]], np.float32)
    recs = [{"id": "a", "path": "a"}, {"id": "b", "path": "b"}, {"id": "c", "path": "c"}]
    out = cosine_rerank(q, cand, recs)
    assert [r.image_id for r in out] == ["b", "c", "a"]


def test_mmr_balances_diversity():
    """With lambda=0, MMR should avoid picking two near-identical candidates first."""
    q = _unit(np.array([1, 0, 0, 0], np.float32))
    cand = np.array([[1, 0.02, 0, 0], [0.99, 0.03, 0, 0],   # near-duplicates
                     [0.5, 0.85, 0, 0]], np.float32)         # diverse
    recs = [{"id": "dup1", "path": "p"}, {"id": "dup2", "path": "p"}, {"id": "div", "path": "p"}]
    picked = [r.image_id for r in mmr_rerank(q, cand, recs, k=2, lambda_param=0.0)]
    assert "div" in picked  # diversity term pulls in the different item


def test_integration_with_metadata_store_and_disk_index(tmp_path):
    """Sprint-2 integration: metadata_store JSON + on-disk FAISS save/load + pipeline.

    Exercises the real cross-module wiring (Data layer metadata_store <-> Pipeline's
    _load_metadata, and FaissIndex.save/load) with a FakeEncoder so no CLIP download
    is needed. Confirms the pipeline reads the Data Engineer's metadata format and
    resolves index ids -> records correctly.
    """
    from src.data.datatypes import ImageRecord
    from src.data.metadata_store import save_metadata

    rng = np.random.default_rng(1)
    c_cat = np.zeros(DIM, np.float32)
    c_cat[0] = 1.0
    c_dog = np.zeros(DIM, np.float32)
    c_dog[1] = 1.0

    embs, records = [], []
    for label, center in (("cat", c_cat), ("dog", c_dog)):
        for j in range(15):
            embs.append(_unit(center + rng.normal(0, 0.05, DIM).astype(np.float32)))
            records.append(ImageRecord(id=f"{label}_{j}", path=f"/img/{label}_{j}.jpg", label=label))
    embeddings = np.vstack(embs).astype(np.float32)

    # Data Engineer's metadata format on disk
    meta_path = tmp_path / "metadata.json"
    save_metadata(records, str(meta_path))

    # FaissIndex save -> load roundtrip (built in the same record order)
    idx_path = tmp_path / "faiss.index"
    built = FaissIndex(dim=DIM, index_type="flat")
    built.build(embeddings)
    built.save(idx_path)
    loaded = FaissIndex(dim=DIM)
    loaded.load(idx_path)
    assert loaded.ntotal == 30

    # pipeline reads metadata via metadata_store; FakeEncoder supplies the query vec
    metadata = RetrievalPipeline._load_metadata(meta_path)
    assert isinstance(metadata, list) and metadata[0]["label"] == "cat"
    pipe = RetrievalPipeline.from_components(FakeEncoder({"cat": c_cat}), loaded, metadata)
    results = pipe.query_by_text("cat", k=5)
    assert len(results) == 5 and all(r.label == "cat" for r in results)
    assert results[0].path.endswith(".jpg")
