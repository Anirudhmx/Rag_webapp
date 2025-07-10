import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from DocumentProcessor import DocumentProcessor
from rag_bot import RAGBot

# ------------------- Page Config -------------------
st.set_page_config(page_title="RAG Bot - Document Q&A", page_icon="🤖", layout="wide")

# ------------------- Sidebar Navigation -------------------
page = st.sidebar.selectbox("Navigate", ["📁 Q&A Interface", "📊 Evaluation Stats"])

# ------------------- Session State -------------------
for key in ['documents', 'embeddings', 'chat_history', 'similarity_scores', 'time_taken', 'llm_choice']:
    if key not in st.session_state:
        st.session_state[key] = [] if key not in ['llm_choice'] else "GPT-2"

# ------------------- Model Selector -------------------
st.sidebar.title("🔧 Model Settings")
llm_options = ["GPT-2", "llama2:7b"]
selected_llm = st.sidebar.selectbox("Choose LLM Model", llm_options)
st.session_state.llm_choice = selected_llm

# ------------------- Upload & Process Documents -------------------
st.sidebar.title("📁 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, TXT or CSV",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

processor = DocumentProcessor()
bot = RAGBot(model_choice=st.session_state.llm_choice)

if uploaded_files:
    start = time.time()
    st.sidebar.write("Processing Files...")
    new_docs, new_embeddings = [], []
    for file in uploaded_files:
        name = file.name
        ext = name.split(".")[-1].lower()
        text = ""
        if ext == "pdf": text = processor.extract_text_from_pdf(file)
        elif ext == "txt": text = processor.extract_text_from_txt(file)
        elif ext == "csv": text = processor.extract_text_from_csv(file)
        if text:
            chunks = processor.chunk_text(text)
            embeddings = processor.create_embeddings(chunks)
            new_docs.append({'name': name, 'type': ext, 'chunks': chunks, 'text_preview': text[:200]})
            new_embeddings.append(embeddings)
    st.session_state.documents = new_docs
    st.session_state.embeddings = new_embeddings
    st.sidebar.success(f"Processed {len(new_docs)} files in {round(time.time() - start, 2)}s")

# ------------------- Q&A Interface -------------------
if page == "📁 Q&A Interface":
    st.title("🤖 RAG Bot Q&A")
    if st.session_state.documents:
        query = st.text_input("Ask a question from the documents")
        if st.button("Ask", type="primary") and query:
            start_time = time.time()
            chunks, sim_scores = bot.find_relevant_chunks(query, st.session_state.documents, st.session_state.embeddings)
            answer = bot.generate_answer(query, chunks)
            end_time = time.time()

            st.session_state.chat_history.append({
                "query": query,
                "answer": answer,
                "chunks": chunks
            })
            st.session_state.similarity_scores.append(sim_scores)
            st.session_state.time_taken.append(end_time - start_time)

            st.write(f"⏱ Time taken: {end_time - start_time:.2f}s")
            st.write("### 💬 Answer")
            st.write(answer)

        if st.session_state.chat_history:
            st.write("### 📜 Previous Queries")
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q: {chat['query']}", expanded=(i == 0)):
                    st.markdown("**Answer:**")
                    st.write(chat['answer'])
                    for j, c in enumerate(chat['chunks']):
                        st.text_area(f"Context {j+1}:", c, height=100, disabled=True)

        if st.button("Clear Chat History"):
            for key in ['chat_history', 'similarity_scores', 'time_taken']:
                st.session_state[key] = []
            st.experimental_rerun()
    else:
        st.info("📤 Upload documents from the sidebar to begin.")

# ------------------- Evaluation Page -------------------
elif page == "📊 Evaluation Stats":
    st.title("📊 Evaluation Metrics")
    if not st.session_state.chat_history:
        st.warning("⚠️ No queries asked yet.")
    else:
        st.write(f"**Total Queries:** {len(st.session_state.chat_history)}")
        st.write(f"**Current LLM:** {st.session_state.llm_choice}")

        avg_sim = np.mean([np.mean(s) for s in st.session_state.similarity_scores])
        st.write(f"**Top-3 Avg Similarity:** {avg_sim:.4f}")
        st.write(f"**Avg Time per Query:** {np.mean(st.session_state.time_taken):.2f}s")

        # Plot similarity scores
        st.subheader("📈 Top-k Similarity per Query")
        fig1, ax1 = plt.subplots()
        for i, sim in enumerate(st.session_state.similarity_scores):
            ax1.plot([1, 2, 3], sim, label=f"Query {i+1}")
        ax1.set_xlabel("Top-k Rank")
        ax1.set_ylabel("Cosine Similarity")
        ax1.legend()
        st.pyplot(fig1)

        # Plot timing stats
        st.subheader("⏱ Time Taken per Query")
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, len(st.session_state.time_taken) + 1), st.session_state.time_taken, marker='o')
        ax2.set_xlabel("Query #")
        ax2.set_ylabel("Time (s)")
        st.pyplot(fig2)
