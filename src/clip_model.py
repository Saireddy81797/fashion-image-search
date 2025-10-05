import torch
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
from PIL import Image


class CLIPModel:
    def __init__(self, device=None):
        """
        Initialize CLIP model using Hugging Face Transformers.
        Compatible with Python 3.13 and Streamlit Cloud.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HFCLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_image_embedding(self, image_path):
        """
        Returns normalized image embedding.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    def get_text_embedding(self, text):
        """
        Returns normalized text embedding.
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()
