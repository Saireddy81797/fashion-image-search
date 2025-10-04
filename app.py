import streamlit as st
import os
from src.clip_model import CLIPModel
from src.search_engine import FashionSearchEngine
from src.diffusion_model import FashionDiffusion
from PIL import Image

st.set_page_config(page_title="Fashion Image Search & GenAI", layout="wide")

clip_model = CLIPModel()
search_engine = FashionSearchEngine(dim=512)
gen_model = FashionDiffusion()

st.title("👗 Fashion Image Search & Generative AI")

option = st.radio("Choose Mode:", ["Image Search", "Text Search", "Generate New Fashion"])

if option == "Image Search":
    uploaded = st.file_uploader("Upload a fashion image", type=["jpg","png"])
    if uploaded:
        query_path = f"temp_{uploaded.name}"
        with open(query_path, "wb") as f:
            f.write(uploaded.getbuffer())

        query_embedding = clip_model.get_image_embedding(query_path)
        _, indices = search_engine.search(query_embedding, top_k=5)

        st.subheader("🔍 Similar Items Found:")
        cols = st.columns(5)
        for i, idx in enumerate(indices[0]):
            img_path = f"data/sample_images/{idx}.jpg"
            cols[i].image(Image.open(img_path), use_container_width=True)

elif option == "Text Search":
    query = st.text_input("Enter your fashion search query:")
    if query:
        query_embedding = clip_model.get_text_embedding(query)
        _, indices = search_engine.search(query_embedding, top_k=5)

        st.subheader("🧵 Matching Fashion Items:")
        cols = st.columns(5)
        for i, idx in enumerate(indices[0]):
            img_path = f"data/sample_images/{idx}.jpg"
            cols[i].image(Image.open(img_path), use_container_width=True)

elif option == "Generate New Fashion":
    prompt = st.text_area("Describe the fashion you want to generate:")
    if st.button("Generate"):
        output = gen_model.generate_variation(prompt)
        st.image(Image.open(output), caption="Generated Fashion", use_container_width=True)
