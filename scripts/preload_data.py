import pandas as pd
import requests
from PIL import Image
import io
import torch
import clip
import chromadb
import numpy as np
import os
import aiohttp
import asyncio

# Load CLIP model
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")

async def embed_image_from_url_async(session, url):
    try:
        async with session.get(url) as response:
            image_data = await response.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_input = clip_preprocess(image).unsqueeze(0)
            with torch.no_grad():
                embedding = clip_model.encode_image(image_input)
            return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error embedding image from {url}: {e}")
        return np.zeros(clip_model.visual.output_dim)

def embed_text_batch(texts):
    text_inputs = clip.tokenize(texts, truncate=True)
    with torch.no_grad():
        embeddings = clip_model.encode_text(text_inputs)
    return embeddings.cpu().numpy()



async def load_and_embed_amazon_products_async(csv_path, chroma_index_folder=None, batch_size=32):
    df = pd.read_csv(csv_path)
    df['id'] = range(len(df))
    df['Image'] = df['Image'].str.split("|").str[0]
    vectors = []
    meta = []
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            # Batch embed text
            names = batch_df["Product Name"].astype(str).tolist()
            descs = batch_df["About Product"].fillna("").astype(str).tolist()
            texts = [f"{name}. {desc}" if desc and desc != "nan" else name for name, desc in zip(names, descs)]
            text_embs = embed_text_batch(texts)

            # Async embed images
            img_urls = batch_df["Image"].tolist()
            tasks = [embed_image_from_url_async(session, url) for url in img_urls]
            img_embs = await asyncio.gather(*tasks)

            for j in range(len(batch_df)):
                combined_emb = np.concatenate([text_embs[j], img_embs[j]])
                vectors.append(combined_emb)
                meta.append({"name": names[j], "description": descs[j], "image_url": img_urls[j], "uniq_id": batch_df.iloc[j]["Uniq Id"]})

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
        
        return collection, meta_df
    
    return None, pd.DataFrame(meta)

if __name__ == "__main__":
    asyncio.run(load_and_embed_amazon_products_async("data/amazon_product_data_2020.csv", "data/chroma_index/"))
