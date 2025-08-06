from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Use a regular dict type for GraphState to allow optional fields
GraphState = Dict[str, Any]


def retrieve_relevant_data(state: GraphState) -> GraphState:
    """Generate a response based on the state."""
    has_image = state.get("has_image", False)
    text_embedding = state['user_embedding']
    image_embedding = state['image_embedding']
    
    # Decide which collection to use
    if has_image:
        collection = state['vector_index']
        metadata_df = state['metadata_df']
        # Squeeze to 1D and then combine
        text_emb_1d = text_embedding.squeeze()
        image_emb_1d = image_embedding.squeeze()
        search_emb = np.concatenate([text_emb_1d, image_emb_1d]).astype("float32")
        logger.info("Using multimodal collection for search")
    elif state.get('text_only_collection') is not None:
        collection = state['text_only_collection']
        metadata_df = state['text_only_metadata']
        search_emb = text_embedding.squeeze().astype("float32")
        logger.info("Using text-only collection for search")
    else:
        # Fallback to multimodal if text-only is not available, but use only text embedding
        collection = state['vector_index']
        metadata_df = state['metadata_df']
        # Here, we need to construct a query that matches the multimodal embedding structure,
        # but with a zero-vector for the image part.
        text_emb_1d = text_embedding.squeeze()
        image_emb_placeholder = np.zeros_like(text_emb_1d) # Assuming image embedding dim is same as text
        search_emb = np.concatenate([text_emb_1d, image_emb_placeholder]).astype("float32")
        logger.warning("Text-only collection not found. Falling back to multimodal search with text query only.")

    # Retrieve relevant documents
    results = collection.query(
        query_embeddings=[search_emb.tolist()],
        n_results=3
    )
    
    # Get indices from the results
    if results['ids'] and len(results['ids'][0]) > 0:
        indices = [int(doc_id) for doc_id in results['ids'][0]]
        source_data = metadata_df.iloc[indices]
    else:
        # Fallback to empty dataframe if no results
        source_data = metadata_df.iloc[:0].copy()

    state['source_data'] = source_data
    state['metadata_df'] = metadata_df  # Ensure metadata_df is updated

    return state


def generate_response(state: GraphState) -> GraphState:
    """Generate a response based on the retrieved documents and optional Amazon data."""
    source_documents = state['source_data']
    amazon_context = state.get('amazon_context', '')
    
    # Build context from local results
    context_list = []
    for _, product in source_documents.iterrows():
        context_str = f"Product Name: {product['name']}\nDescription: {product['description']}"
        if 'image_url' in product and pd.notna(product['image_url']):
             context_str += f"\nImage URL: {product['image_url']}"
        context_list.append(context_str)
    
    local_context = "\n\n---\n\n".join(context_list)
    
    # Combine local and Amazon context
    full_context = local_context + amazon_context
    
    response = state['response_chain'].invoke({
        "query": state['user_input'],
        "context": full_context,
    })
    
    state['response'] = response
    return state



def create_enhanced_workflow() -> StateGraph:
    """Create an enhanced workflow for the RAG system."""
    
    response_builder = StateGraph(GraphState)
    response_builder.add_node("retrieve_relevant_data", retrieve_relevant_data)
    response_builder.add_node("generate_response", generate_response)

    response_builder.add_edge(START, "retrieve_relevant_data")
    response_builder.add_edge("retrieve_relevant_data", "generate_response")
    response_builder.add_edge("generate_response", END)

    response_workflow = response_builder.compile()

    return response_workflow

