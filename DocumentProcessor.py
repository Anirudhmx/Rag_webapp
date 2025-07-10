import PyPDF2
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer
import streamlit as st
class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_text_from_pdf(self, uploaded_file) -> str:
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
        try:
            return str(uploaded_file.read(), "utf-8")
        except Exception as e:
            st.error(f"Error reading TXT file: {str(e)}")
            return ""

    def extract_text_from_csv(self, uploaded_file) -> str:
        try:
            df = pd.read_csv(uploaded_file)
            return df.to_string(index=False)
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
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

    def create_embeddings(self, chunks: List[str]):
        return self.model.encode(chunks)