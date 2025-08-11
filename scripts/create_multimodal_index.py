import pandas as pd
import requests
from PIL import Image
import io
import torch
from conv_ai_ecommerce.utils.load_utils import get_clip_model
import chromadb
import numpy as np
import os
import aiohttp
import asyncio

from conv_ai_ecommerce.utils.load_utils import get_clip_model, generate_text_clip_embeddings, generate_image_embeddings

# Load CLIP model
clip_model, clip_preprocess = get_clip_model()

async def load_and_embed_amazon_products_async(csv_path, chroma_index_folder=None, batch_size=32):
    df = pd.read_csv(csv_path)
    vectors = []
    meta = []
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            # Batch embed text
            
            texts = batch_df['search_text'].to_list()
            text_embs = generate_text_clip_embeddings(texts, clip_model, clip_preprocess)

            # Async embed images
            img_urls = batch_df["image_url"].tolist()
            img_embs = await generate_image_embeddings(session, img_urls, clip_model, clip_preprocess)

            for j in range(len(batch_df)):
                combined_emb = np.concatenate([text_embs[j], img_embs[j]])
                vectors.append(combined_emb)
                meta.append({"name": batch_df.iloc[j]["name"], "image_url": img_urls[j], "product_id": batch_df.iloc[j]["product_id"], "search_text": batch_df.iloc[j].get("search_text", "")})

    vectors_np = np.stack(vectors).astype("float32")
    
    # Create Chroma collection
    if chroma_index_folder:
        os.makedirs(chroma_index_folder, exist_ok=True)
        client = chromadb.PersistentClient(path=chroma_index_folder)
        
        # Delete collection if it exists (for fresh start)
        try:
            client.delete_collection("amazon_products")
        except:
            pass
            
        # Create collection with explicit L2 distance metric
        collection = client.create_collection(
            name="amazon_products",
            metadata={"hnsw:space": "l2"}
        )
        
        # Add embeddings to collection
        ids = [str(i) for i in range(len(vectors_np))]
        embeddings = vectors_np.tolist()
        metadatas = meta
        
        collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        meta_df = pd.DataFrame(meta)
        meta_df.to_csv(chroma_index_folder + "/index_metadata.csv")

        print(f"Text-only collection created successfully!")
        print(f"   - {len(vectors_np)} products indexed")
        print(f"   - Embedding dimension: {vectors_np.shape[1]}")
        print(f"   - Saved to: {chroma_index_folder}")
        
        return collection, meta_df
    
    return None, pd.DataFrame(meta)

if __name__ == "__main__":
    asyncio.run(load_and_embed_amazon_products_async("data/amazon_products_cleaned.csv", "data/chroma_index/"))
