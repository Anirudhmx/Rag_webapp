from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from DocumentProcessor import DocumentProcessor
import numpy as np
import streamlit as st
import torch
import subprocess

class RAGBot:
    def __init__(self, model_choice="GPT-2"):
        self.model_choice = model_choice
        self.processor = DocumentProcessor()
        self.llm_model, self.llm_tokenizer = self.load_model()

    def load_model(self):
        if self.model_choice == "GPT-2":
            try:
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                model = GPT2LMHeadModel.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
                return model, tokenizer
            except Exception as e:
                st.error(f"Error loading GPT-2: {e}")
                return None, None
        else:
            return None, None  # For Ollama models, not needed here

    def find_relevant_chunks(self, query: str, documents, embeddings, top_k: int = 3):
        if not documents or not embeddings:
            return [], []
        query_embedding = self.processor.model.encode([query])
        all_embeddings = np.vstack(embeddings)
        all_chunks = [chunk for doc in documents for chunk in doc['chunks']]
        similarities = cosine_similarity(query_embedding, all_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [all_chunks[i] for i in top_indices], [similarities[i] for i in top_indices]

    def generate_answer(self, query: str, context_chunks: list) -> str:
        context_text = "\n".join(context_chunks[:2])
        prompt = f"Based on the following context, answer the question:\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

        if self.model_choice == "GPT-2":
            if self.llm_model is None or self.llm_tokenizer is None:
                return "Error loading GPT-2 model."
            try:
                inputs = self.llm_tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
                outputs = self.llm_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    attention_mask=torch.ones(inputs.shape, dtype=torch.long)
                )
                generated_text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_start = generated_text.find("Answer:") + len("Answer:")
                return generated_text[answer_start:].strip()[:300] + "..."
            except Exception as e:
                st.error(f"Error generating with GPT-2: {str(e)}")
                return "Error generating answer using GPT-2."

        else:
            # Use Ollama for external models
            try:
                result = subprocess.run(
                    ["ollama", "run", self.model_choice.lower()],
                    input=prompt.encode("utf-8"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60
                )
                output = result.stdout.decode().strip()
                if not output:
                    return "Ollama model did not return any output."
                return output.split("Answer:")[-1].strip() if "Answer:" in output else output
            except subprocess.TimeoutExpired:
                return "Ollama model took too long to respond."
            except Exception as e:
                return f"Error using Ollama model '{self.model_choice}': {e}"

