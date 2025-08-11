import streamlit as st
from PIL import Image
import os
import numpy as np
import chromadb
import pandas as pd
import torch
import clip
from conv_ai_ecommerce.utils.load_utils import get_text_only_model, generate_image_embedding_from_file, generate_text_clip_embeddings, generate_text_only_embeddings, get_clip_model, generate_image_embeddings
from conv_ai_ecommerce.utils.chroma_utils import load_all_collections, load_chroma_index
from conv_ai_ecommerce.utils.response_utils import retrieve_relevant_products, generate_response



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
        collections = load_all_collections("data/chroma_index")
        multimodal_collection = collections['multimodal']
        text_only_collection = collections['text_only']
        image_only_collection = collections['image_only']
        
    except Exception as e:
        # Fallback to regular loading
        st.warning(f"Could not load all collections, using multimodal only: {e}")
        multimodal_collection = load_chroma_index("data/chroma_index")
        text_only_collection = None
        image_only_collection = None
    
    clip_model, clip_processor = get_clip_model()
    text_model = get_text_only_model()
    
    return (multimodal_collection,
            text_only_collection, 
            image_only_collection, 
            clip_model, clip_processor, text_model)

(collection, text_only_collection, image_only_collection, 
 clip_model, clip_processor, text_model) = load_data()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []



user_input = st.text_input("Ask about a product or get recommendations:", key="user_input")
image_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

# Add Send button, and only show 'Send as a Follow-Up' if chat history exists
if st.session_state["chat_history"]:
    col1, col2 = st.columns([1, 1])
    send_clicked = col1.button("Send")
    followup_clicked = col2.button("Send as a Follow-Up")
else:
    send_clicked = st.button("Send")
    followup_clicked = False

def handle_send(clear_history: bool):
    if user_input.strip() or image_file:
        if clear_history:
            # Clear chat history for fresh start
            st.session_state["chat_history"] = []
        # Add current query to history
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        if user_input.strip():
            # Embed user query
            if image_file:
                query_emb = generate_text_clip_embeddings(user_input, clip_model=clip_model, clip_processor=clip_processor)
            else:
                query_emb = generate_text_only_embeddings(user_input, text_model=text_model)
        else:
            query_emb = None
        # Embed image if present
        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", width=200)
            # Save uploaded image temporarily
            temp_path = "temp_uploaded_image.png"
            image.save(temp_path)
            # Preprocess the image and compute the embedding
            image_emb = generate_image_embedding_from_file(image, clip_model=clip_model, clip_preprocess=clip_processor)
            os.remove(temp_path)
        else:
            image_emb = None
        retrieved_items = retrieve_relevant_products(collection, text_only_collection, image_only_collection, query_emb, image_emb)
        response = generate_response(user_input, retrieved_items)
        print(response)
        st.session_state["chat_history"].append({"role": "bot", "content": response})
        st.rerun()

if send_clicked:
    handle_send(clear_history=True)
elif followup_clicked:
    handle_send(clear_history=False)

# Show chat history if it exists
if st.session_state["chat_history"]:
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user" and msg['content']:
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Amazon Assistant:** {msg['content']}")

