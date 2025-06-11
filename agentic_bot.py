import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
from transformers import pipeline

st.set_page_config(page_title="📘 Free AI Tutor (Summarized)", layout="wide")
st.title("📘 Free AI Tutor: Management & Leadership (Summarized Answers)")

@st.cache_resource
def load_text_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    chunk_size = 500
    overlap = 100
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    return chunks

@st.cache_resource
def build_vector_store(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors, chunks, model

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

chunks = load_text_chunks("management_book.pdf")
index, vectors, raw_chunks, model = build_vector_store(chunks)
summarizer = load_summarizer()

query = st.chat_input("Ask me anything about the course...")
if query:
    q_vector = model.encode([query])
    scores, idxs = index.search(q_vector, k=3)
    results = [raw_chunks[i] for i in idxs[0]]

    combined = " ".join(results)
    summary = summarizer(combined, max_length=200, min_length=50, do_sample=False)[0]['summary_text']

    st.chat_message("user").markdown(query)
    st.chat_message("assistant").markdown(f"📘 **Summary based on your textbook:**\n\n{summary}")
