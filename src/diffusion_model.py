from diffusers import StableDiffusionPipeline
import torch

class FashionDiffusion:
    def __init__(self):
        model_id = "stabilityai/stable-diffusion-2-1-base"  # lighter than 2-1
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_variation(self, prompt):
        image = self.pipe(prompt).images[0]
        output_path = "generated_fashion.png"
        image.save(output_path)
        return output_path
