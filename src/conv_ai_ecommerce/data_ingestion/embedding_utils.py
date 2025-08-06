import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

def create_search_embedding(
    text_embedding: np.ndarray, 
    image_embedding: np.ndarray, 
    has_image: bool = False,
    strategy: str = "text_only_when_no_image"
) -> np.ndarray:
    """
    Create search embedding based on whether image is present.
    
    Args:
        text_embedding: Text embedding from CLIP (shape: 1, 512 or 512,)
        image_embedding: Image embedding from CLIP (shape: 1, 512 or 512,)
        has_image: Whether an image was actually provided by the user
        strategy: Strategy for handling missing images
            - "text_only_when_no_image": Use only text embedding when no image
            - "zero_image": Use zero image embedding (current behavior)
            - "masked_multimodal": Use weighted combination
    
    Returns:
        np.ndarray: Search embedding ready for vector database query
    """
    # Ensure embeddings are 1D
    text_emb_1d = text_embedding.squeeze()
    image_emb_1d = image_embedding.squeeze()
    
    if strategy == "text_only_when_no_image" and not has_image:
        # Return only text embedding for text-only search
        logger.debug("Using text-only search (no image provided)")
        return text_emb_1d.astype("float32")
    
    elif strategy == "masked_multimodal" and not has_image:
        # Use text embedding + small random noise in image dimensions
        # This prevents zero vectors from dominating similarity
        logger.debug("Using masked multimodal search")
        noise_scale = 0.01
        random_image = np.random.normal(0, noise_scale, image_emb_1d.shape).astype("float32")
        return np.concatenate([text_emb_1d, random_image]).astype("float32")
    
    else:
        # Default: concatenate text and image embeddings
        if has_image:
            logger.debug("Using full multimodal search (image provided)")
        else:
            logger.debug("Using zero-padded multimodal search")
        return np.concatenate([text_emb_1d, image_emb_1d]).astype("float32")

def search_with_fallback(
    collection,
    text_embedding: np.ndarray,
    image_embedding: np.ndarray,
    has_image: bool,
    n_results: int = 3,
    strategies: list = None
) -> Tuple[Dict[str, Any], str]:
    """
    Search vector database with fallback strategies.
    
    Args:
        collection: ChromaDB collection
        text_embedding: Text embedding
        image_embedding: Image embedding  
        has_image: Whether image was provided
        n_results: Number of results to return
        strategies: List of strategies to try in order
    
    Returns:
        Tuple of (search_results, strategy_used)
    """
    if strategies is None:
        if has_image:
            strategies = ["zero_image"]  # Use standard multimodal search
        else:
            strategies = ["text_only_when_no_image", "zero_image"]
    
    for strategy in strategies:
        try:
            search_emb = create_search_embedding(
                text_embedding, 
                image_embedding, 
                has_image, 
                strategy
            )
            
            # Query the collection
            results = collection.query(
                query_embeddings=[search_emb.tolist()],
                n_results=n_results,
                include=['metadatas', 'distances']
            )
            
            # Check if we got meaningful results
            if results['ids'] and len(results['ids'][0]) > 0:
                logger.info(f"Search successful with strategy: {strategy}")
                return results, strategy
            
        except Exception as e:
            logger.warning(f"Search failed with strategy {strategy}: {e}")
            continue
    
    # If all strategies fail, return empty results
    logger.error("All search strategies failed")
    return {
        'ids': [[]],
        'distances': [[]],
        'metadatas': [[]]
    }, "failed"


