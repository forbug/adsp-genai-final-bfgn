import torch
from PIL import Image
import clip
import os

def get_clip_model(device="cpu"):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def compute_clip_embedding(image_path, device="cpu"):
    model, preprocess = get_clip_model(device)
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
    return embedding.cpu().numpy()

# Example usage:
# embedding = compute_clip_embedding("path/to/image.jpg")

