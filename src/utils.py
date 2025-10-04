import os
from PIL import Image

def load_images_from_folder(folder):
    images = []
    paths = []
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, filename)
            try:
                img = Image.open(path)
                images.append(img)
                paths.append(path)
            except:
                continue
    return images, paths
