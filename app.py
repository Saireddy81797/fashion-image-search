import streamlit as st
import os
from PIL import Image

# ------------------------------
# Lazy imports for performance
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    try:
        from src.clip_model import CLIPModel
        from src.search_engine import FashionSearchEngine
        from src.diffusion_model import FashionDiffusion

        clip_model = CLIPModel()
        search_engine = FashionSearchEngine()
        gen_model = FashionDiffusion()

        st.success("✅ Models loaded successfully!")
        return clip_model, search_engine, gen_model, True

    except Exception as e:
        st.warning(f"⚠️ Model loading failed: {e}")
        return None, None, None, False


st.set_page_config(page_title="Fashion Image Search & GenAI", layout="wide")

st.title("👗 Fashion Image Search & Generative AI")
st.caption("Hosted by **Sai Reddy** 💫")

# ------------------------------
# Load models (cached)
# ------------------------------
clip_model, search_engine, gen_model, MODEL_AVAILABLE = load_models()

# ------------------------------
# Mode selection
# ------------------------------
option = st.radio("Choose Mode:", ["Image Search", "Text Search", "Generate New Fashion"])

# ------------------------------
# Image Search
# ------------------------------
if option == "Image Search":
    uploaded = st.file_uploader("Upload a fashion image", type=["jpg", "png"])
    if uploaded:
        query_path = f"temp_{uploaded.name}"
        with open(query_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if MODEL_AVAILABLE:
            query_embedding = clip_model.get_image_embedding(query_path)
            results, _ = search_engine.search(query_embedding, top_k=5)

            st.subheader("🔍 Similar Items Found:")
            cols = st.columns(len(results))
            for i, img_file in enumerate(results):
                img_path = f"data/sample_images/{img_file}"
                if os.path.exists(img_path):
                    cols[i].image(Image.open(img_path), use_container_width=True)
                else:
                    cols[i].warning("Image missing")
        else:
            st.warning("Running in demo mode. Models not available on this environment.")
            st.image("data/sample_images/demo1.jpg", caption="Demo Fashion Image")

# ------------------------------
# Text Search
# ------------------------------
elif option == "Text Search":
    query = st.text_input("Enter your fashion search query:")
    if query:
        if MODEL_AVAILABLE:
            query_embedding = clip_model.get_text_embedding(query)
            results, _ = search_engine.search(query_embedding, top_k=5)

            st.subheader("🧵 Matching Fashion Items:")
            cols = st.columns(len(results))
            for i, img_file in enumerate(results):
                img_path = f"data/sample_images/{img_file}"
                if os.path.exists(img_path):
                    cols[i].image(Image.open(img_path), use_container_width=True)
        else:
            st.warning("Running in demo mode. Models not available on this environment.")
            st.image("data/sample_images/demo2.jpg", caption="Demo Search Result")

# ------------------------------
# Fashion Generation
# ------------------------------
elif option == "Generate New Fashion":
    prompt = st.text_area("Describe the fashion you want to generate:")
    if st.button("Generate"):
        if MODEL_AVAILABLE:
            output = gen_model.generate_variation(prompt)
            st.image(Image.open(output), caption="Generated Fashion", use_container_width=True)
        else:
            st.warning("Model not available. Showing demo output.")
            st.image("data/sample_images/demo_generated.jpg", caption="Demo Generated Fashion")
