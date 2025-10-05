import streamlit as st
from PIL import Image
import os

# ✅ Fix for OMP duplicate library error (important for PyTorch / Hugging Face)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ✅ Health check log for Streamlit Cloud
print("✅ Streamlit app started successfully!")

st.set_page_config(page_title="Fashion Image Search & GenAI", layout="wide")
st.title("👗 Fashion Image Search & Generative AI")
st.caption("Hosted by **Sai Reddy** 💫")

st.info("⚡ Running in demo mode for Streamlit Cloud (models disabled due to size limit)")

option = st.radio("Choose Mode:", ["Image Search", "Text Search", "Generate New Fashion"])

if option == "Image Search":
    uploaded = st.file_uploader("Upload a fashion image", type=["jpg", "png"])
    if uploaded:
        st.subheader("🔍 Similar Items Found:")
        cols = st.columns(5)
        for i, img_name in enumerate(["demo1.jpg", "demo2.jpg", "demo_generated.jpg", "demo1.jpg", "demo2.jpg"]):
            img_path = os.path.join("data/sample_images", img_name)
            cols[i].image(Image.open(img_path), use_container_width=True)

elif option == "Text Search":
    query = st.text_input("Enter your fashion search query:")
    if query:
        st.subheader("🧵 Matching Fashion Items:")
        cols = st.columns(3)
        for i, img_name in enumerate(["demo1.jpg", "demo2.jpg", "demo_generated.jpg"]):
            img_path = os.path.join("data/sample_images", img_name)
            cols[i].image(Image.open(img_path), use_container_width=True)

elif option == "Generate New Fashion":
    prompt = st.text_area("Describe the fashion you want to generate:")
    if st.button("Generate"):
        st.image("data/sample_images/demo_generated.jpg", caption="Generated Fashion", use_container_width=True)
