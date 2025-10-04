import os
from PIL import Image

class FashionDataset:
    def __init__(self, folder):
        self.folder = folder
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder) 
                            if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return Image.open(self.image_paths[idx]), self.image_paths[idx]
