import torch
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
from PIL import Image

class CLIPModel:
    def __init__(self, device=None):
        try:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            # ✅ Use smaller and faster CLIP model
            model_name = "openai/clip-vit-base-patch32"
            self.model = HFCLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)

            self.available = True
        except Exception as e:
            print(f"[WARNING] CLIP model could not be loaded: {e}")
            self.available = False

    def get_image_embedding(self, image_path):
        """Extract normalized CLIP embedding for an image."""
        if not self.available:
            raise RuntimeError("CLIP model not available")

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    def get_text_embedding(self, text):
        """Extract normalized CLIP embedding for a text query."""
        if not self.available:
            raise RuntimeError("CLIP model not available")

        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()
