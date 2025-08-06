import streamlit as st
from PIL import Image
import os
import numpy as np
import chromadb
import pandas as pd
import torch
import clip
from conv_ai_ecommerce.data_ingestion.chroma_loader import load_chroma_index, load_dual_collections
from conv_ai_ecommerce.data_ingestion.clip_embeddings import compute_clip_embedding, get_clip_model
from conv_ai_ecommerce.vlrag_framework.graph import create_enhanced_workflow
from conv_ai_ecommerce.vlrag_framework.prompts import create_response_chain, base_system_prompt


st.set_page_config(page_title="Amazon Product Chatbot", page_icon="🛒", layout="centered")
st.markdown("""
    <style>
        .reportview-container {
            background: #fff;
        }
        .sidebar .sidebar-content {
            background: #232f3e;
        }
        .stButton>button {
            color: #fff;
            background-color: #ff9900;
            border-radius: 20px;
        }
        .stTextInput>div>input {
            border-radius: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
st.title("Amazon Product Chatbot")
st.write("Chat interface for product recommendations and Q&A.")

# Load Chroma index and metadata (cache for session)
@st.cache_resource
def load_data():
    # Try to load both collections
    try:
        collections = load_dual_collections("data/chroma_index")
        multimodal_collection = collections['multimodal']['collection']
        multimodal_metadata = collections['multimodal']['metadata']
        text_only_collection = collections['text_only']['collection']
        text_only_metadata = collections['text_only']['metadata']
        
        if text_only_collection is not None:
            st.info("💡 Text-only collection loaded - will use optimized text search when no image is provided")
    except Exception as e:
        # Fallback to regular loading
        st.warning(f"Could not load dual collections, using multimodal only: {e}")
        multimodal_collection, multimodal_metadata = load_chroma_index("data/chroma_index")
        text_only_collection, text_only_metadata = None, None
    
    clip_model, _ = clip.load("ViT-B/32", device="cpu")
    workflow = create_enhanced_workflow()
    response_chain = create_response_chain()
    
    return (multimodal_collection, multimodal_metadata, 
            text_only_collection, text_only_metadata, 
            clip_model, workflow, response_chain)

(collection, meta_df, text_only_collection, text_only_metadata, 
 clip_model, workflow, response_chain) = load_data()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Ask about a product or get recommendations:", key="user_input")
image_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

if st.button("Send"):
    if user_input.strip():
        # Clear chat history for fresh start
        st.session_state["chat_history"] = []
        
        # Add current query to history
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        
        # Embed user query
        text_input = clip.tokenize([user_input], truncate=True)
        with torch.no_grad():
            query_emb = clip_model.encode_text(text_input).cpu().numpy()

        # Embed image if present
        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", width=200)
            # Save uploaded image temporarily
            temp_path = "temp_uploaded_image.png"
            image.save(temp_path)
            # Preprocess the image and compute the embedding
            image_input = clip.load("ViT-B/32", device="cpu")[1](image).unsqueeze(0)
            with torch.no_grad():
                image_emb = clip_model.encode_image(image_input).cpu().numpy()
            os.remove(temp_path)
        else:
            image_emb = np.zeros_like(query_emb)

        initial_state = {
            "user_input": user_input,
            "user_embedding": query_emb,
            "image_embedding": image_emb,
            "has_image": image_file is not None,  # Track if image was provided
            "vector_index": collection,
            "metadata_df": meta_df,
            "text_only_collection": text_only_collection,  # Add text-only collection
            "text_only_metadata": text_only_metadata,      # Add text-only metadata
            "response_chain": response_chain,
        }

        response = workflow.invoke(initial_state)

        assistant_content = response['response']
        
        st.session_state["chat_history"].append({"role": "assistant", "content": assistant_content})
        
        # Force page refresh by rerunning
        st.rerun()

for msg in st.session_state["chat_history"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

