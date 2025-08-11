#!/usr/bin/env python3

"""
Create a text-only ChromaDB collection for better text-only search.
This allows us to have separate indices optimized for text-only vs multimodal queries.
"""

import pandas as pd
import torch
import chromadb
import numpy as np
import os
import asyncio

from conv_ai_ecommerce.utils.load_utils import get_text_only_model, generate_text_embeddings


text_model = get_text_only_model()

def create_text_only_collection(csv_path: str, chroma_index_folder: str, batch_size: int = 32):
    """Create a text-only ChromaDB collection."""
    
    print("Loading product data...")
    df = pd.read_csv(csv_path)
    
    vectors = []
    meta = []
    
    print(f"Processing {len(df)} products in batches of {batch_size}...")
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        # Create text combinations (same as multimodal version)
        texts = batch_df['search_text'].to_list()
        
        # Embed text only
        text_embs = generate_text_embeddings(texts, text_model)
        
        for j in range(len(batch_df)):
            # Store only text embeddings (512 dimensions instead of 1024)
            vectors.append(text_embs[j])
            meta.append({
                "name": batch_df.iloc[j]["name"], 
                "image_url": batch_df.iloc[j].get("image_url", ""), 
                "product_id": batch_df.iloc[j]["product_id"],
                "search_text": batch_df.iloc[j].get("search_text", ""), 
            })
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + batch_size}/{len(df)} products...")
    
    vectors_np = np.stack(vectors).astype("float32")
    
    print("Creating ChromaDB collection...")
    os.makedirs(chroma_index_folder, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_index_folder)
    
    # Delete collection if it exists
    try:
        client.delete_collection("amazon_products_text_only")
    except:
        pass
    
    # Create text-only collection
    collection = client.create_collection(
        name="amazon_products_text_only",
        metadata={"hnsw:space": "cosine"}  # Cosine similarity for text
    )
    
    # Add embeddings to collection
    ids = [str(i) for i in range(len(vectors_np))]
    embeddings = vectors_np.tolist()
    metadatas = meta
    
    print("Adding embeddings to collection...")
    collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    # Save metadata
    meta_df = pd.DataFrame(meta)
    meta_df.to_csv(chroma_index_folder + "/index_metadata.csv", index=False)
    
    print(f"Text-only collection created successfully!")
    print(f"   - {len(vectors_np)} products indexed")
    print(f"   - Embedding dimension: {vectors_np.shape[1]}")
    print(f"   - Saved to: {chroma_index_folder}")
    
    return collection, meta_df

if __name__ == "__main__":
    create_text_only_collection(
        "data/amazon_products_cleaned.csv", 
        "data/chroma_index_text_only/"
    )
