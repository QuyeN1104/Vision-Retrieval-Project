"""
app.py — Streamlit entry point for Vision Retrieval.
Owner: UI Engineer (Sprint 1 skeleton → Sprint 2 pipeline integration)

Sprint 2 tasks completed:
  UI-1: Tích hợp RetrievalPipeline vào Streamlit
  UI-2: render_result_card() — thumbnail, similarity score, label, path
  UI-3: render_sidebar() — Top-K, score threshold, label filter dropdown
  UI-4: query_by_image() — file_uploader → encode → search → display
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ─── Page config (must be first Streamlit call) ─────────────────────

st.set_page_config(
    page_title="Vision Retrieval Project",
    page_icon="🔎",
    layout="wide",
)


# ─── Pipeline loader (cached so CLIP only loads once) ───────────────

@st.cache_resource
def load_pipeline():
    """Load the retrieval pipeline once and cache across reruns."""
    from src.pipeline.retrieval_pipeline import RetrievalPipeline
    return RetrievalPipeline()


# ─── UI-2: Result card ──────────────────────────────────────────────

def render_result_card(result, col):
    """
    Render a single result card: thumbnail, similarity score, label, path.

    Args:
        result: RetrievalResult namedtuple with .path, .score, .label, .image_id
        col: Streamlit column to render into.
    """
    with col:
        img_path = result.path
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Image not found: {img_path}")

        score_color = "normal" if result.score >= 0.3 else "off"
        st.metric("Similarity", f"{result.score:.4f}", delta_color=score_color)

        if result.label:
            st.caption(f"🏷️ **{result.label}**")
        st.caption(f"📁 `{Path(img_path).name}`")


# ─── UI-3: Results gallery ──────────────────────────────────────────

def render_results_gallery(results):
    """Display results in a responsive 3-column grid."""
    if not results:
        st.warning("No results found matching your query.")
        return

    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, result in enumerate(results[i : i + cols_per_row]):
            render_result_card(result, cols[j])


# ─── Score chart (uses Sprint 1 UI-4 visualize helper) ──────────────

def render_score_chart(results):
    """Plot similarity score distribution for the returned results."""
    from src.utils.visualize import draw_score_bar

    scores = [r.score for r in results]
    ids = [Path(r.path).stem for r in results]
    fig = draw_score_bar(scores, ids, top_k=len(results))
    st.pyplot(fig)
    plt.close(fig)


# ─── Main App ────────────────────────────────────────────────────────

st.title("🔎 Vision Retrieval Project")
st.caption("Image retrieval powered by CLIP + FAISS")

# Check if index exists before loading
from config import get_config
cfg = get_config()
index_path = Path(cfg.INDEX_PATH)

if not index_path.exists():
    st.error(
        f"⚠️ FAISS index not found at `{index_path}`.\n\n"
        "To run with the sample dataset, execute:\n\n"
        "```bash\n"
        "uv run python scripts/build_index.py --data-dir data/sample_images --output-dir data/sample_index\n"
        "```\n\n"
        "Or update `.env` to point to your real index."
    )
    st.stop()

# Load pipeline (cached)
with st.spinner("Loading CLIP model and FAISS index..."):
    pipeline = load_pipeline()

st.success(f"✅ Pipeline ready — **{pipeline.index.ntotal}** images indexed", icon="🚀")

# ─── UI-3: Sidebar — Top-K, score threshold, label filter ───────────

with st.sidebar:
    st.header("⚙️ Search Settings")

    query_mode = st.radio("Query type", ["Text", "Image"], horizontal=True)

    top_k = st.slider("Top K results", min_value=1, max_value=20, value=5)

    score_threshold = st.slider(
        "Min similarity score", min_value=0.0, max_value=1.0, value=0.0, step=0.01
    )

    # Build label filter from metadata
    labels = sorted(set(r.get("label") for r in pipeline.metadata if r.get("label")))
    label_options = ["All"] + labels
    selected_label = st.selectbox("Filter by label", label_options)
    label_filter = None if selected_label == "All" else selected_label

    st.divider()
    st.caption(f"Index: {pipeline.index.ntotal} vectors")
    st.caption(f"Labels: {len(labels)} unique")


# ─── Search area ─────────────────────────────────────────────────────

st.subheader("Search")

if query_mode == "Text":
    # ── UI-1: Text query integration ──
    query_text = st.text_input(
        "Enter text query",
        placeholder="Example: a person standing near a river",
    )
    search_clicked = st.button("🔍 Search", type="primary")

    if search_clicked:
        if not query_text.strip():
            st.warning("Please enter a query before searching.")
        else:
            with st.spinner("Encoding query and searching..."):
                results = pipeline.query_by_text(
                    query_text, k=top_k, label_filter=label_filter
                )
                # Apply score threshold
                results = [r for r in results if r.score >= score_threshold]

            st.subheader(f"Top {len(results)} Results")
            render_results_gallery(results)

            if results:
                st.subheader("📊 Score Analysis")
                render_score_chart(results)

else:
    # ── UI-4: Image query integration ──
    if "query_image_state" not in st.session_state:
        st.session_state.query_image_state = None

    uploaded_file = st.file_uploader(
        "Upload query image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file:
        st.session_state.query_image_state = Image.open(uploaded_file).convert("RGB")

    st.markdown("**Or choose a sample image:**")
    sample_images = [
        {"name": "a person standing near a river", "path": "data/test_image/191.jpg"},
        {"name": "a boat on the water", "path": "data/test_image/079.jpg"},
    ]
    
    cols = st.columns(len(sample_images))
    for idx, sample in enumerate(sample_images):
        with cols[idx]:
            if os.path.exists(sample["path"]):
                st.image(sample["path"], caption=sample["name"], width=200)
                if st.button("Use this image", key=f"btn_sample_{idx}"):
                    st.session_state.query_image_state = Image.open(sample["path"]).convert("RGB")
            else:
                st.warning(f"Sample not found: {sample['path']}")

    query_image = st.session_state.query_image_state

    if query_image:
        st.markdown("---")
        st.image(query_image, caption="Selected Query Image", width=300)

    if query_image:
        search_clicked = st.button("🔍 Search by Image", type="primary")

        if search_clicked:
            with st.spinner("Encoding image and searching..."):
                results = pipeline.query_by_image(
                    query_image, k=top_k, label_filter=label_filter
                )
                results = [r for r in results if r.score >= score_threshold]

            st.subheader(f"Top {len(results)} Results")
            render_results_gallery(results)

            if results:
                st.subheader("📊 Score Analysis")
                render_score_chart(results)