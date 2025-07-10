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


class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_text_from_pdf(self, uploaded_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, uploaded_file) -> str:
        """Extract text from TXT file"""
        try:
            return str(uploaded_file.read(), "utf-8")
        except Exception as e:
            st.error(f"Error reading TXT file: {str(e)}")
            return ""
    
    def extract_text_from_csv(self, uploaded_file) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(uploaded_file)
            # Convert DataFrame to string representation
            text = df.to_string(index=False)
            return text
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks with overlap"""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        return self.model.encode(chunks)