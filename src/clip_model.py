import torch
import clip
from PIL import Image

class CLIPModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def get_image_embedding(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(image).cpu().numpy()

    def get_text_embedding(self, text):
        text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(text).cpu().numpy()
