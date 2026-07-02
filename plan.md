# Vision Retrieval Project — AGENT.md
> **Image Retrieval System using CLIP + FAISS + Streamlit**
> Last updated: 2026-06-15

---

## 1. Project Overview

Hệ thống truy vấn hình ảnh (Image Retrieval) cho phép tìm kiếm ảnh tương đồng từ một tập dữ liệu ảnh đã chuẩn bị sẵn. Người dùng có thể tìm bằng **text query** hoặc **ảnh upload**, hệ thống trả về top-K ảnh liên quan nhất.

**Luồng hoạt động chính:**
```
Input (text / image)
      │
      ▼
 CLIP Encoder  ──►  Query Embedding (512-d)
      │
      ▼
 FAISS Index  ──►  Top-K Nearest Neighbors (ANN Search)
      │
      ▼
 Result Reranker (optional: cosine re-score)
      │
      ▼
 Streamlit UI  ──►  Hiển thị ảnh kết quả + similarity score
```

---

## 2. Architecture

### 2.1 System Layers

```
┌──────────────────────────────────────────────────────────┐
│                    Layer 4: UI                           │
│              Streamlit App (app.py)                      │
│   Text Query | Image Upload | Results Gallery | Filters  │
└──────────────────────┬───────────────────────────────────┘
                       │  Python function calls
┌──────────────────────▼───────────────────────────────────┐
│                  Layer 3: Pipeline                       │
│   retrieval_pipeline.py — orchestrate query → results    │
│   reranker.py — cosine re-score, diversity filter        │
└────────────┬──────────────────────┬──────────────────────┘
             │                      │
┌────────────▼──────────┐  ┌────────▼─────────────────────┐
│    Layer 2: Search    │  │    Layer 2: Encoding          │
│    faiss_index.py     │  │    clip_encoder.py            │
│    ANN search, filter │  │    Text + Image → embedding   │
└────────────┬──────────┘  └────────┬─────────────────────┘
             │                      │
┌────────────▼──────────────────────▼──────────────────────┐
│                  Layer 1: Data                           │
│   dataset_loader.py — load images, metadata              │
│   indexer.py — build/save/load FAISS index               │
│   metadata_store.py — image paths, labels, captions      │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│                  Layer 0: Foundation                     │
│   config.py | logger.py | utils.py | metrics.py          │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Cấu trúc thư mục

```
vision-retrieval-project/
├── app.py                        # Streamlit entry point
├── config.py                     # Centralized config (paths, model, top-k)
├── src/
│   ├── data/
│   │   ├── dataset_loader.py     # Load ảnh từ thư mục / CSV manifest
│   │   ├── metadata_store.py     # Lưu/đọc metadata (path, label, caption)
│   │   └── preprocessor.py       # Resize, normalize ảnh trước khi encode
│   ├── model/
│   │   ├── clip_encoder.py       # CLIP text/image encoder wrapper
│   │   └── embeddings_cache.py   # Cache embeddings đã tính (numpy .npy)
│   ├── search/
│   │   ├── faiss_index.py        # Build / save / load / query FAISS index
│   │   └── reranker.py           # Cosine re-score, MMR diversity rerank
│   ├── pipeline/
│   │   ├── retrieval_pipeline.py # Orchestrate: encode → search → rerank → format
│   │   ├── indexer.py            # Offline: encode dataset → build FAISS index
│   │   └── batch_encode.py       # Batch encoding large datasets efficiently
│   └── utils/
│       ├── logger.py             # Structured logging
│       ├── metrics.py            # Recall@K, mAP, MRR calculation
│       └── visualize.py          # Draw bounding boxes, plot score distribution
├── scripts/
│   ├── build_index.py            # CLI: python scripts/build_index.py --data-dir ...
│   ├── eval_retrieval.py         # Evaluate Recall@K, mAP on labeled test set
│   └── bench_latency.py          # Benchmark query latency (p50/p95/p99)
├── tests/
│   ├── unit/
│   │   ├── test_clip_encoder.py
│   │   ├── test_faiss_index.py
│   │   └── test_reranker.py
│   └── integration/
│       └── test_pipeline.py
├── data/
│   ├── images/                   # Thư mục ảnh gốc (do Data Engineer chuẩn bị)
│   ├── index/                    # FAISS index files (.faiss, .json)
│   └── metadata.json             # {id, path, label, caption, split}
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_clip_analysis.ipynb
│   └── 03_evaluation.ipynb
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 3. Team Roles

| Role | Trách nhiệm chính |
|------|------------------|
| **Tech Lead (Nguyễn Phúc Định Quyền)** | Kiến trúc tổng thể, code review, integration, CI/CD, tài liệu |
| **Data Engineer(Nguyễn Trần Ngọc Thịnh)** | Chuẩn bị dataset, metadata, preprocessing, test set annotation |
| **Model Engineer(Kiều Hoàng Quân)** | CLIP encoder, embedding cache, model experiments |
| **Pipeline Engineer(Lê Đình Minh Quân)** | FAISS index, retrieval pipeline, reranker, batch encoding |
| **UI Engineer(Trần Phước Anh Khoa)** | Streamlit app, UX, kết nối UI ↔ pipeline, visualization |

---

## 4. Sprint Plan

> **3 sprints × 1 tuần.** Mỗi thành viên có **4 tasks/sprint** cụ thể đến cấp độ hàm.

---

### 🏃 Sprint 1 — Foundation: Data + Model + Indexing (Tuần 1)

**Mục tiêu:** Dataset sẵn sàng, CLIP encode được ảnh, FAISS index đầu tiên chạy được.

#### 🔵 Tech Lead
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| TL-1 | `config.py` | `Config` dataclass: `DATA_DIR`, `INDEX_PATH`, `MODEL_NAME`, `TOP_K`, `DEVICE` |
| TL-2 | `src/utils/logger.py` | `get_logger(name)` — structured logging với level, timestamp |
| TL-3 | `pyproject.toml` / `requirements.txt` | Thiết lập dependencies: `torch`, `clip`, `faiss-cpu`, `streamlit`, `Pillow` |
| TL-4 | `README.md` | Hướng dẫn setup, cài đặt, chạy index, chạy app |

---

#### 🟢 Data Engineer
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| DE-1 | `src/data/dataset_loader.py` | `load_dataset(data_dir) -> List[ImageRecord]` — scan thư mục ảnh, trả về danh sách path + metadata |
| DE-2 | `src/data/dataset_loader.py` | `load_from_manifest(csv_path) -> List[ImageRecord]` — load từ CSV (path, label, caption, split) |
| DE-3 | `src/data/metadata_store.py` | `save_metadata(records, path)`, `load_metadata(path) -> List[ImageRecord]` — lưu/đọc `metadata.json` |
| DE-4 | `src/data/preprocessor.py` | `preprocess_image(img_path) -> PIL.Image` — resize, convert RGB, validate định dạng |

**Deliverable:** `data/metadata.json` với ≥ 1000 ảnh đã có label.

---

#### 🟡 Model Engineer
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| ME-1 | `src/model/clip_encoder.py` | `CLIPEncoder.__init__(model_name, device)` — load model `openai/clip-vit-base-patch32` |
| ME-2 | `src/model/clip_encoder.py` | `encode_text(text: str) -> np.ndarray` — tokenize → forward → normalize L2 |
| ME-3 | `src/model/clip_encoder.py` | `encode_image(image: PIL.Image) -> np.ndarray` — preprocess → forward → normalize L2 |
| ME-4 | `src/model/clip_encoder.py` | `encode_images_batch(images: List, batch_size=32) -> np.ndarray` — batch encode efficient |

---

#### 🟠 Pipeline Engineer
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| PE-1 | `src/search/faiss_index.py` | `FaissIndex.__init__(dim, index_type)` — khởi tạo `IndexFlatIP` hoặc `IndexIVFFlat` |
| PE-2 | `src/search/faiss_index.py` | `build(embeddings: np.ndarray)` — train + add vectors |
| PE-3 | `src/search/faiss_index.py` | `save(path)` / `load(path)` — serialize/deserialize index |
| PE-4 | `src/pipeline/batch_encode.py` | `encode_dataset(loader, encoder, batch_size) -> np.ndarray` — encode toàn bộ dataset thành embeddings matrix |

---

#### 🔴 UI Engineer
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| UI-1 | `app.py` | Skeleton Streamlit app: sidebar config, main layout, page title |
| UI-2 | `app.py` | `render_search_input()` — tab Text Search / Image Upload |
| UI-3 | `app.py` | `render_results_gallery(results)` — grid ảnh với score badge |
| UI-4 | `src/utils/visualize.py` | `draw_score_bar(scores)` — plot similarity score distribution (matplotlib) |

---

### 🏃 Sprint 2 — Core Pipeline: Search + Retrieval + UI Integration (Tuần 2)

**Mục tiêu:** End-to-end query chạy được — nhập text/ảnh → ra kết quả trên Streamlit.

#### 🔵 Tech Lead
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| TL-1 | `scripts/build_index.py` | CLI script: `argparse` → load data → encode → build FAISS → save |
| TL-2 | `src/pipeline/indexer.py` | `Indexer.run(data_dir, output_dir)` — orchestrate toàn bộ offline indexing flow |
| TL-3 | `tests/unit/test_faiss_index.py` | Unit tests: build, save/load, search correctness |
| TL-4 | GitHub Actions | `.github/workflows/test.yml` — `pytest` + `ruff` lint on push |

---

#### 🟢 Data Engineer
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| DE-1 | `data/` | Hoàn thiện dataset: clean ảnh lỗi, đảm bảo metadata đầy đủ |
| DE-2 | `src/data/dataset_loader.py` | `split_dataset(records, test_ratio=0.1)` — tách train/test set |
| DE-3 | `data/test_queries.json` | Tạo thủ công **50 query–ground_truth pairs** cho evaluation (text query → list ảnh đúng) |
| DE-4 | `notebooks/01_data_exploration.ipynb` | EDA: phân phối class, số ảnh/category, sample visualization |

---

#### 🟡 Model Engineer
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| ME-1 | `src/model/embeddings_cache.py` | `EmbeddingsCache.save(embeddings, ids, path)` — lưu `.npy` + id mapping |
| ME-2 | `src/model/embeddings_cache.py` | `EmbeddingsCache.load(path) -> (np.ndarray, List[str])` — load lại cache |
| ME-3 | `src/model/clip_encoder.py` | `encode_image_from_path(path) -> np.ndarray` — convenience wrapper |
| ME-4 | `notebooks/02_clip_analysis.ipynb` | Visualize embedding space (UMAP/t-SNE), phân tích cluster by class |

---

#### 🟠 Pipeline Engineer
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| PE-1 | `src/search/faiss_index.py` | `search(query_vec, k) -> (distances, indices)` — ANN search |
| PE-2 | `src/search/faiss_index.py` | `filter_search(query_vec, k, label_filter)` — search với metadata filter |
| PE-3 | `src/search/reranker.py` | `cosine_rerank(query_vec, candidates, metadata) -> List[RetrievalResult]` — re-score chính xác |
| PE-4 | `src/pipeline/retrieval_pipeline.py` | `RetrievalPipeline.query_by_text(text, k) -> List[RetrievalResult]` |

---

#### 🔴 UI Engineer
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| UI-1 | `app.py` | Tích hợp `RetrievalPipeline` vào Streamlit — kết nối input → pipeline → display |
| UI-2 | `app.py` | `render_result_card(result)` — thumbnail, similarity score, label, path |
| UI-3 | `app.py` | `render_sidebar()` — sliders: Top-K, score threshold, label filter dropdown |
| UI-4 | `app.py` | `query_by_image()` — `st.file_uploader` → encode → search → display |

---

### 🏃 Sprint 3 — Evaluation, Reranking & Polish (Tuần 3)

**Mục tiêu:** Đánh giá benchmark đầy đủ, tối ưu retrieval, UI hoàn thiện, demo-ready.

#### 🔵 Tech Lead
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| TL-1 | `scripts/eval_retrieval.py` | `evaluate(pipeline, test_queries, k_list=[1,5,10])` — tính Recall@K, mAP, MRR |
| TL-2 | `scripts/bench_latency.py` | Benchmark: text query latency, image query latency (p50/p95/p99, N=200 runs) |
| TL-3 | `docs/results.md` | Báo cáo kết quả benchmark, so sánh các cấu hình FAISS |
| TL-4 | Final review | Code review toàn bộ PR, merge, tag release `v1.0.0` |

---

#### 🟢 Data Engineer
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| DE-1 | `scripts/eval_retrieval.py` | Cung cấp `data/test_queries.json` hoàn chỉnh + ground truth labels |
| DE-2 | `src/utils/metrics.py` | `recall_at_k(retrieved, relevant, k) -> float` |
| DE-3 | `src/utils/metrics.py` | `average_precision(retrieved, relevant) -> float`, `mean_ap(results) -> float` |
| DE-4 | `notebooks/03_evaluation.ipynb` | Visualization kết quả eval: precision-recall curve, failure cases |

---

#### 🟡 Model Engineer
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| ME-1 | `src/model/clip_encoder.py` | Thử nghiệm model lớn hơn: `clip-vit-large-patch14`, so sánh accuracy vs latency |
| ME-2 | `src/utils/metrics.py` | `mrr_at_k(retrieved_list, relevant_set, k) -> float` |
| ME-3 | `tests/unit/test_clip_encoder.py` | Unit tests: shape check, L2 norm, text/image consistency |
| ME-4 | `docs/model_comparison.md` | Bảng so sánh: ViT-B/32 vs ViT-L/14 — Recall@5, mAP, latency |

---

#### 🟠 Pipeline Engineer
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| PE-1 | `src/pipeline/retrieval_pipeline.py` | `query_by_image(image, k) -> List[RetrievalResult]` — image query flow |
| PE-2 | `src/search/reranker.py` | `mmr_rerank(query_vec, candidates, lambda_=0.5) -> List[RetrievalResult]` — Maximum Marginal Relevance để tăng diversity |
| PE-3 | `src/search/faiss_index.py` | Thử nghiệm `IndexIVFFlat` (nlist=100) — so sánh với `IndexFlatIP` về speed/recall |
| PE-4 | `tests/integration/test_pipeline.py` | Integration test: text query → top5 kết quả có label đúng |

---

#### 🔴 UI Engineer
| # | File | Hàm / Nhiệm vụ |
|---|------|----------------|
| UI-1 | `app.py` | History panel: lưu 5 query gần nhất trong `st.session_state` |
| UI-2 | `app.py` | `render_eval_tab()` — tab debug: hiển thị Recall@5, mAP realtime trên test set |
| UI-3 | `app.py` | Responsive layout: 2-column (sidebar config + main results) |
| UI-4 | `app.py` | Export kết quả: nút tải ZIP ảnh kết quả, CSV scores |

---

## 5. Metrics & Benchmarks

### 5.1 Retrieval Quality

| Metric | Mô tả | Target | Đo bởi |
|--------|--------|--------|--------|
| **Recall@1** | Top-1 kết quả đúng | ≥ 0.50 | `metrics.recall_at_k(..., k=1)` |
| **Recall@5** | Có đúng trong top-5 | ≥ 0.75 | `metrics.recall_at_k(..., k=5)` |
| **Recall@10** | Có đúng trong top-10 | ≥ 0.85 | `metrics.recall_at_k(..., k=10)` |
| **mAP@10** | Mean Average Precision | ≥ 0.60 | `metrics.mean_ap(...)` |
| **MRR@10** | Mean Reciprocal Rank | ≥ 0.60 | `metrics.mrr_at_k(..., k=10)` |

**Test set:** 50 query–ground_truth pairs, do Data Engineer annotate thủ công.

### 5.2 Latency

| Metric | Mô tả | Target |
|--------|--------|--------|
| **Text query p50** | Median encode + FAISS search | ≤ 200ms |
| **Text query p95** | 95th percentile | ≤ 500ms |
| **Image query p50** | Upload + encode + FAISS search | ≤ 400ms |
| **Index build time** | Encode 10K ảnh + build FAISS | ≤ 10 phút (CPU) |

### 5.3 Index Quality

| Metric | Mô tả | Target |
|--------|--------|--------|
| **Index size** | Kích thước file FAISS | ≤ 500MB / 100K ảnh |
| **IVF recall loss** | Recall drop khi dùng IVFFlat vs FlatIP | ≤ 3% |

### 5.4 Cách đo (Protocol)

```python
# scripts/eval_retrieval.py
from src.utils.metrics import recall_at_k, mean_ap, mrr_at_k
from src.pipeline.retrieval_pipeline import RetrievalPipeline
import json

def evaluate(pipeline: RetrievalPipeline, test_path: str, k_list=[1, 5, 10]):
    queries = json.load(open(test_path))  # [{query, relevant_ids: []}]
    results = {f"Recall@{k}": [] for k in k_list}
    ap_scores, mrr_scores = [], []

    for item in queries:
        retrieved = pipeline.query_by_text(item["query"], k=max(k_list))
        retrieved_ids = [r.image_id for r in retrieved]
        relevant = set(item["relevant_ids"])

        for k in k_list:
            results[f"Recall@{k}"].append(recall_at_k(retrieved_ids, relevant, k))
        ap_scores.append(average_precision(retrieved_ids, relevant))
        mrr_scores.append(mrr_at_k(retrieved_ids, relevant, max(k_list)))

    return {k: sum(v)/len(v) for k, v in results.items()} | {
        "mAP": sum(ap_scores) / len(ap_scores),
        "MRR@10": sum(mrr_scores) / len(mrr_scores),
    }
```

---

## 6. Workload Summary

| Role | Sprint 1 | Sprint 2 | Sprint 3 | Total |
|------|----------|----------|----------|-------|
| **Tech Lead** | 4 | 4 | 4 | **12** |
| **Data Engineer** | 4 | 4 | 4 | **12** |
| **Model Engineer** | 4 | 4 | 4 | **12** |
| **Pipeline Engineer** | 4 | 4 | 4 | **12** |
| **UI Engineer** | 4 | 4 | 4 | **12** |
| **Total** | **20** | **20** | **20** | **60** |

---

## 7. Task Dependencies

```
Sprint 1 (song song):
  TL-1 (config.py)  ──► TẤT CẢ tasks khác
  DE-1,2,3,4        ──► cung cấp data cho PE, ME
  ME-1,2,3,4        ──► cung cấp CLIPEncoder cho PE, UI
  PE-1,2,3,4        ──► cung cấp FaissIndex cho Pipeline
  UI-1,2,3,4        ──► skeleton app, chưa cần pipeline

Sprint 2 (có dependency):
  PE-4 (retrieval_pipeline) REQUIRES ME-1,2,3 + PE-1,2
  UI-1 (integration)        REQUIRES PE-4
  DE-3 (test set)           REQUIRED BY TL-1 (Sprint 3 eval)

Sprint 3 (sequential):
  TL-1 (eval script)        REQUIRES DE-1,2,3 + PE hoàn chỉnh
  PE-2 (MMR rerank)         REQUIRES PE-1 (query_by_image)
  ME-1 (large model test)   REQUIRES ME baseline đã chạy
```

---

## 8. Tech Stack

| Component | Technology | Lý do |
|-----------|-----------|-------|
| Embedding | `openai/clip-vit-base-patch32` (HuggingFace) | Multimodal text+image, zero-shot mạnh |
| Vector Search | FAISS (`faiss-cpu` / `faiss-gpu`) | Nhanh, offline, không cần server |
| UI | Streamlit | Đơn giản, Python-native, đủ cho demo |
| Image Processing | Pillow, torchvision | Standard |
| Evaluation | Custom scripts + numpy | Recall@K, mAP, MRR |
| Testing | pytest | Unit + integration |
| Lint/Format | ruff | Fast Python linter |

---

## 9. Definition of Done

Task được coi là **Done** khi:
- [ ] Code reviewed qua Pull Request (ít nhất 1 người approve)
- [ ] Unit tests pass (`pytest tests/unit/`)
- [ ] Không có lỗi lint (`ruff check src/`)
- [ ] Docstring đầy đủ cho public functions/classes
- [ ] Merge vào branch `develop`
- [ ] Benchmark target của sprint được đáp ứng (nếu applicable)

---

*Cập nhật sau mỗi sprint retrospective.*
