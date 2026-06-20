import random

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageDraw


st.set_page_config(
    page_title="Vision Retrieval Project",
    page_icon="🔎",
    layout="wide",
)


def create_dummy_image(label, score):
    image = Image.new("RGB", (480, 270), color=(34, 40, 49))
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), label, fill=(255, 255, 255))
    draw.text((24, 58), f"Similarity: {score:.2f}", fill=(126, 231, 135))
    return image


def mock_search_images(query_text, top_k):
    results = []

    for index in range(top_k):
        score = round(random.uniform(0.55, 0.96), 2)
        results.append(
            {
                "image": create_dummy_image(f"Frame #{index + 1}", score),
                "image_path": f"data/frames/vid1_{index + 1:03d}.jpg",
                "score": score,
                "video_path": "data/raw_videos/vid1.mp4",
                "timestamp": (index + 1) * 5,
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)


def render_gallery(results):
    columns = st.columns(3)

    for index, item in enumerate(results):
        with columns[index % 3]:
            st.image(item["image"], use_container_width=True)
            st.caption(item["image_path"])
            st.metric("Similarity", f"{item['score']:.2f}")
            st.write(f"Timestamp: {item['timestamp']}s")


def render_score_chart(results):
    scores = [item["score"] for item in results]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(scores, bins=8, color="#4F8CFF", edgecolor="white")
    ax.set_title("Similarity Score Distribution")
    ax.set_xlabel("Similarity score")
    ax.set_ylabel("Number of frames")
    ax.set_xlim(0, 1)

    st.pyplot(fig)


st.title("Vision Retrieval Project")
st.caption("Sprint 1 UI mockup using dummy retrieval results.")

with st.sidebar:
    st.header("Search Settings")
    query_mode = st.radio("Query type", ["Text", "Image"], horizontal=True)
    top_k = st.slider("Top K results", min_value=1, max_value=20, value=5)

st.subheader("Search")

query_text = ""

if query_mode == "Text":
    query_text = st.text_input(
        "Enter text query",
        placeholder="Example: a blue shirt",
    )
else:
    uploaded_file = st.file_uploader(
        "Upload query image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Query image", width=300)

search_clicked = st.button("Search", type="primary")

if search_clicked:
    if query_mode == "Text" and not query_text.strip():
        st.warning("Please enter a query before searching.")
    else:
        results = mock_search_images(query_text, top_k)

        st.subheader("Top Results")
        render_gallery(results)

        st.subheader("Score Analysis")
        render_score_chart(results)
else:
    st.info("Enter a query and click Search to preview Sprint 1 mock results.")