import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

st.set_page_config(page_title="📘 Free AI Tutor", layout="wide")
st.title("📘 Free AI Tutor: Management & Leadership")

@st.cache_resource
def load_text_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    chunk_size = 700
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

chunks = load_text_chunks("management_book.pdf")
index, vectors, raw_chunks, model = build_vector_store(chunks)

query = st.chat_input("Ask me anything about the course...")
if query:
    q_vector = model.encode([query])
    scores, idxs = index.search(q_vector, k=3)
    results = [raw_chunks[i] for i in idxs[0]]
    answer = "**Based on your textbook, here's what I found:**\n\n"
answer += "\n---\n".join(results)
for word in query.lower().split():
    answer = answer.replace(word, f"**{word}**")
st.chat_message("user").markdown(query)
    st.chat_message("assistant").markdown(answer)


