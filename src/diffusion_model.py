import torch
from diffusers import StableDiffusionPipeline

class FashionDiffusion:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_variation(self, prompt, save_path="output.png"):
        image = self.pipe(prompt).images[0]
        image.save(save_path)
        return save_path

