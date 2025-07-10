import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import streamlit as st
import PyPDF2
from typing import List, Dict, Any
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from DocumentProcessor import DocumentProcessor


class RAGBot:
    def __init__(self):
        self.load_gpt2_model()
    
    @st.cache_resource
    def load_gpt2_model(_self):
        """Load GPT-2 model and tokenizer"""
        try:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Add padding token
            tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
        except Exception as e:
            st.error(f"Error loading GPT-2 model: {str(e)}")
            return None, None
    
    def find_relevant_chunks(self, query: str, documents: List[Dict], embeddings: List[np.ndarray], top_k: int = 3) -> List[str]:
        """Find most relevant document chunks for the query"""
        if not documents or not embeddings:
            return []
        
        # Create embedding for query
        processor = DocumentProcessor()
        query_embedding = processor.model.encode([query])
        
        # Combine all embeddings
        all_embeddings = np.vstack(embeddings)
        all_chunks = []
        
        for doc in documents:
            all_chunks.extend(doc['chunks'])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, all_embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_chunks = [all_chunks[i] for i in top_indices]
        
        return relevant_chunks
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer using GPT-2"""
        if not context:
            return "I couldn't find relevant information in the uploaded documents to answer your question."
        
        model, tokenizer = self.load_gpt2_model()
        if model is None or tokenizer is None:
            return "Error: Could not load GPT-2 model."
        
        # Prepare context
        context_text = "\n".join(context[:2])  # Use top 2 chunks to avoid token limit
        
        # Create prompt
        prompt = f"Based on the following context, answer the question:\n\nContext: {context_text[:800]}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            # Tokenize input
            inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,  # Add 150 tokens for the answer
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones(inputs.shape, dtype=torch.long)
                )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part
            answer_start = generated_text.find("Answer:") + len("Answer:")
            answer = generated_text[answer_start:].strip()
            
            # Clean up the answer
            if len(answer) > 300:
                answer = answer[:300] + "..."
            
            return answer if answer else "I couldn't generate a proper answer based on the context."
            
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return f"Based on the context, here's the most relevant information: {context_text[:500]}..."
