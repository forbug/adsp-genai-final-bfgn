#!/usr/bin/env python3

"""
Test script to compare text-only vs multimodal search performance.
This demonstrates how masking image dimensions improves retrieval for text-only queries.
"""

import numpy as np
import clip
import torch
import logging
from conv_ai_ecommerce.data_ingestion.chroma_loader import load_chroma_index, load_dual_collections
from conv_ai_ecommerce.vlrag_framework.graph import create_enhanced_workflow
from conv_ai_ecommerce.vlrag_framework.prompts import create_response_chain

# Enable debug logging to see which collection is used
logging.basicConfig(level=logging.INFO)

def compare_search_strategies():
    """Compare text-only vs multimodal search strategies."""
    
    print("🔧 Loading components...")
    
    # Load both collections
    try:
        collections = load_dual_collections("data/chroma_index")
        multimodal_collection = collections['multimodal']['collection']
        multimodal_metadata = collections['multimodal']['metadata']
        text_only_collection = collections['text_only']['collection']
        text_only_metadata = collections['text_only']['metadata']
        print("✅ Both collections loaded successfully")
    except Exception as e:
        print(f"❌ Error loading collections: {e}")
        return
    
    clip_model, _ = clip.load('ViT-B/32', device='cpu')
    workflow = create_enhanced_workflow()
    response_chain = create_response_chain()
    print("✅ Models and workflow loaded\n")
    
    # Test queries
    test_queries = [
        "gaming headset for xbox",
        "laptop for programming", 
        "wireless bluetooth earbuds",
        "chess board game"
    ]
    
    for query in test_queries:
        print(f"🔍 Testing query: '{query}'")
        print("=" * 60)
        
        # Create embeddings
        text_input = clip.tokenize([query], truncate=True)
        with torch.no_grad():
            query_emb = clip_model.encode_text(text_input).cpu().numpy()
        image_emb = np.zeros_like(query_emb)
        
        # Test 1: Without text-only collection (multimodal search with zero image)
        print("\n📊 Test 1: Multimodal search (with zero image embedding)")
        state_multimodal = {
            'user_input': query,
            'user_embedding': query_emb,
            'image_embedding': image_emb,
            'has_image': False,
            'vector_index': multimodal_collection,
            'metadata_df': multimodal_metadata,
            'text_only_collection': None,  # Force multimodal search
            'text_only_metadata': None,
            'response_chain': response_chain,
        }
        
        response_multimodal = workflow.invoke(state_multimodal)
        print("📋 Multimodal Results:")
        if len(response_multimodal['source_data']) > 0:
            for i, (_, product) in enumerate(response_multimodal['source_data'].iterrows()):
                print(f"  {i+1}. {product['name']}")
        else:
            print("  No results found")
        
        # Test 2: With text-only collection
        print("\n📊 Test 2: Text-only search (512D embeddings)")
        state_text_only = {
            'user_input': query,
            'user_embedding': query_emb,
            'image_embedding': image_emb,
            'has_image': False,
            'vector_index': multimodal_collection,
            'metadata_df': multimodal_metadata,
            'text_only_collection': text_only_collection,
            'text_only_metadata': text_only_metadata,
            'response_chain': response_chain,
        }
        
        response_text_only = workflow.invoke(state_text_only)
        print("📋 Text-Only Results:")
        if len(response_text_only['source_data']) > 0:
            for i, (_, product) in enumerate(response_text_only['source_data'].iterrows()):
                print(f"  {i+1}. {product['name']}")
        else:
            print("  No results found")
        
        # Compare results
        multimodal_names = set(response_multimodal['source_data']['name'].tolist())
        text_only_names = set(response_text_only['source_data']['name'].tolist())
        
        print(f"\n🔍 Analysis:")
        print(f"  • Multimodal search found: {len(multimodal_names)} products")
        print(f"  • Text-only search found: {len(text_only_names)} products")
        print(f"  • Overlap: {len(multimodal_names.intersection(text_only_names))} products")
        print(f"  • Text-only unique: {len(text_only_names - multimodal_names)} products")
        print(f"  • MCP used (multimodal): {response_multimodal.get('used_mcp', False)}")
        print(f"  • MCP used (text-only): {response_text_only.get('used_mcp', False)}")
        
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    compare_search_strategies()
