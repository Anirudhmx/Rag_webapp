import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import streamlit as st
import pandas as pd
import PyPDF2
from typing import List, Dict, Any
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from DocumentProcessor import DocumentProcessor
from RAGBot import RAGBot

# Page configuration
st.set_page_config(
    page_title="RAG Bot - Document Q&A",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'gpt2_model' not in st.session_state:
    st.session_state.gpt2_model = None
if 'gpt2_tokenizer' not in st.session_state:
    st.session_state.gpt2_tokenizer = None

# Initialize components
processor = DocumentProcessor()
bot = RAGBot()

# Sidebar for file upload and settings
st.sidebar.title("📁 Document Upload")

# Model loading status
with st.sidebar:
    st.write("### 🤖 Model Status")
    model, tokenizer = bot.load_gpt2_model()
    if model is not None:
        st.success("✅ GPT-2 model loaded successfully!")
    else:
        st.error("❌ Failed to load GPT-2 model")

# File upload
uploaded_files = st.sidebar.file_uploader(
    "Upload your documents",
    type=['pdf', 'txt', 'csv'],
    accept_multiple_files=True,
    help="Upload PDF, TXT, or CSV files"
)

# Process uploaded files
if uploaded_files:
    with st.sidebar:
        st.write("### Processing Files...")
        progress_bar = st.progress(0)
        
        new_documents = []
        new_embeddings = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            file_type = file_name.split('.')[-1].lower()
            
            # Extract text based on file type
            if file_type == 'pdf':
                text = processor.extract_text_from_pdf(uploaded_file)
            elif file_type == 'txt':
                text = processor.extract_text_from_txt(uploaded_file)
            elif file_type == 'csv':
                text = processor.extract_text_from_csv(uploaded_file)
            else:
                st.error(f"Unsupported file type: {file_type}")
                continue
            
            if text:
                # Chunk the text
                chunks = processor.chunk_text(text)
                
                if chunks:
                    # Create embeddings
                    embeddings = processor.create_embeddings(chunks)
                    
                    # Store document info
                    doc_info = {
                        'name': file_name,
                        'type': file_type,
                        'chunks': chunks,
                        'text_preview': text[:200] + "..." if len(text) > 200 else text
                    }
                    
                    new_documents.append(doc_info)
                    new_embeddings.append(embeddings)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Update session state
        st.session_state.documents = new_documents
        st.session_state.embeddings = new_embeddings
        
        st.success(f"✅ Processed {len(new_documents)} documents!")

# Display uploaded documents
if st.session_state.documents:
    st.sidebar.write("### 📚 Uploaded Documents")
    for doc in st.session_state.documents:
        with st.sidebar.expander(f"📄 {doc['name']}"):
            st.write(f"**Type:** {doc['type'].upper()}")
            st.write(f"**Chunks:** {len(doc['chunks'])}")
            st.write(f"**Preview:** {doc['text_preview']}")

# Main interface
st.title("🤖 RAG Bot - Document Q&A")
st.markdown("Upload your documents and ask questions about their content!")

# Chat interface
if st.session_state.documents:
    # Query input
    query = st.text_input("💬 Ask a question about your documents:", 
                         placeholder="What is the main topic discussed in the documents?")
    
    if st.button("🔍 Ask", type="primary") and query:
        with st.spinner("Searching for relevant information..."):
            # Find relevant chunks
            relevant_chunks = bot.find_relevant_chunks(
                query, 
                st.session_state.documents, 
                st.session_state.embeddings
            )
            
            # Generate answer
            answer = bot.generate_answer(query, relevant_chunks)
            
            # Add to chat history
            st.session_state.chat_history.append({
                'question': query,
                'answer': answer,
                'relevant_chunks': relevant_chunks
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.write("## 💬 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question']}", expanded=(i==0)):
                st.write("**Answer:**")
                st.write(chat['answer'])
                
                if chat['relevant_chunks']:
                    st.write("**📋 Source Context:**")
                    for j, chunk in enumerate(chat['relevant_chunks'][:2]):  # Show top 2 chunks
                        st.text_area(f"Relevant excerpt {j+1}:", chunk, height=100, disabled=True)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

else:
    # Welcome message
    st.info("👆 Please upload your documents using the sidebar to get started!")
    
    st.write("## How to use:")
    st.write("1. Upload PDF, TXT, or CSV files using the sidebar")
    st.write("2. Wait for GPT-2 model to load (first time only)")
    st.write("3. Ask questions about your documents")
    st.write("4. Get AI-generated answers based on document content")
    
    st.write("## Supported file types:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("📄 **PDF** - Text extraction from PDF documents")
    with col2:
        st.write("📝 **TXT** - Plain text files")
    with col3:
        st.write("📊 **CSV** - Tabular data files")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🎈 | Powered by GPT-2 & Sentence Transformers")
st.markdown("**Models used:** GPT-2 (Text Generation) | all-MiniLM-L6-v2 (Embeddings) - Both are free and open-source!")
