import requests
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
import io
from sentence_transformers import SentenceTransformer

def get_text_only_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")  # or "bge-base-en"
    return model

def get_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, preprocess

def generate_text_only_embeddings(texts, text_model):
    """Generate text-only embeddings using all-MiniLM."""
    return text_model.encode(texts, normalize_embeddings=True)

def generate_text_clip_embeddings(texts, clip_model, clip_processor):
    text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        return clip_model.get_text_features(**text_inputs).cpu().numpy()

def preprocess_image(image, clip_processor):
    """Preprocess an image for CLIP."""
    return clip_processor(images=image, return_tensors="pt")


def generate_image_embedding_from_file(image_obj, clip_model, clip_preprocess):
    """Generate an image embedding from a file."""
    image = image_obj.convert("RGB")
    image_input = preprocess_image(image, clip_processor=clip_preprocess)
    with torch.no_grad():
        return clip_model.get_image_features(**image_input).cpu().numpy()


async def generate_image_embeddings(session, image_urls, clip_model, clip_preprocess):
    """Generate image embeddings using CLIP."""
    embeddings = []
    for image_url in image_urls:
        try:
            async with session.get(image_url) as response:
                image_data = await response.read()
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image_input = preprocess_image(image, clip_processor=clip_preprocess)
                with torch.no_grad():
                    embeddings.append(clip_model.get_image_features(**image_input).cpu().numpy())
        except Exception as e:
            print(f"Error embedding image from {image_url}: {e}")
            embeddings.append(np.zeros(clip_model.config.projection_dim))
    return np.vstack(embeddings)

# Helper function to download files
def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return save_path
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download {url}: {e}")