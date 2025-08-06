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
    return collection, metadata

def load_text_only_collection(index_path) -> Tuple[Optional[object], Optional[pd.DataFrame]]:
    """Load text-only ChromaDB collection if it exists."""
    try:
        client = chromadb.PersistentClient(path=index_path)
        collection = client.get_collection("amazon_products_text_only")
        metadata_path = os.path.join(index_path, "text_only_metadata.csv")
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
        else:
            # Fallback to regular metadata if text-only metadata doesn't exist
            metadata = pd.read_csv(os.path.join(index_path, "index_metadata.csv"))
        return collection, metadata
    except Exception as e:
        print(f"Text-only collection not found: {e}")
        return None, None

def load_dual_collections(multimodal_path, text_only_path=None):
    """Load both multimodal and text-only collections."""
    # Load multimodal collection
    multimodal_collection, multimodal_metadata = load_chroma_index(multimodal_path)
    
    # Try to load text-only collection
    if text_only_path is None:
        text_only_path = multimodal_path.replace("chroma_index", "chroma_index_text_only")
    
    text_only_collection, text_only_metadata = load_text_only_collection(text_only_path)
    
    return {
        'multimodal': {'collection': multimodal_collection, 'metadata': multimodal_metadata},
        'text_only': {'collection': text_only_collection, 'metadata': text_only_metadata}
    }
