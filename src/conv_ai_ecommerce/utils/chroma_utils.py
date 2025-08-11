import chromadb
import pandas as pd
import os
from chromadb.config import Settings
from typing import Tuple, Optional

def load_chroma_index(index_path):
    """Load Chroma vector database from the specified path."""
    client = chromadb.PersistentClient(path=index_path)
    collection = client.get_collection("amazon_products")
    metadata = pd.read_csv(index_path + "/index_metadata.csv")
    return collection

def load_text_only_collection(index_path) -> Tuple[Optional[object], Optional[pd.DataFrame]]:
    """Load text-only ChromaDB collection if it exists."""
    try:
        client = chromadb.PersistentClient(path=index_path)
        collection = client.get_collection("amazon_products_text_only")
    
        return collection
    except Exception as e:
        print(f"Text-only collection not found: {e}")
        return None
    

def load_image_only_collection(index_path):
    """Load image-only ChromaDB collection if it exists."""
    try:
        client = chromadb.PersistentClient(path=index_path)
        collection = client.get_collection("amazon_products_image_only")
        return collection
    except Exception as e:
        print(f"Image-only collection not found: {e}")
        return None
    

def load_all_collections(multimodal_path, text_only_path=None, image_only_path=None):
    """Load both multimodal and text-only and image-only collections."""
    # Load multimodal collection
    multimodal_collection = load_chroma_index(multimodal_path)
    
    # Try to load text-only collection
    if text_only_path is None:
        text_only_path = multimodal_path.replace("chroma_index", "chroma_index_text_only")
    text_only_collection = load_text_only_collection(text_only_path)

    # Try to load text-only collection
    if image_only_path is None:
        image_only_path = multimodal_path.replace("chroma_index", "chroma_index_image_only")
    image_only_collection = load_image_only_collection(image_only_path)
    
    return {
        'multimodal': multimodal_collection,
        'text_only': text_only_collection,
        'image_only': image_only_collection,
    }
