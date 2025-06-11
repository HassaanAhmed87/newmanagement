
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

st.set_page_config(page_title="Leadership AI Tutor", layout="wide")
st.title("📘 AI Tutor: Management & Leadership")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("management_book.pdf")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embedding = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embedding)

vectorstore = load_vectorstore()
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me anything about the course...")
if user_input:
    with st.spinner("Thinking..."):
        result = qa_chain.run({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history.append((user_input, result))

for q, a in st.session_state.chat_history:
    st.chat_message("user").markdown(q)
    st.chat_message("assistant").markdown(a)
