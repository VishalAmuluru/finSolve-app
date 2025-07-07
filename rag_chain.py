import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# ✅ Set API Key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def load_vectorstore():
    """Load or build FAISS vectorstore with OpenAI embeddings."""
    embeddings = OpenAIEmbeddings()

    # If index exists, load it
    if os.path.exists("faiss_index/index.faiss"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # If not, create one from a sample file (data.txt)
    if not os.path.exists("data.txt"):
        raise FileNotFoundError("❌ 'data.txt' not found. Please include it in the repo.")

    loader = TextLoader("data.txt", encoding="utf-8")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def get_qa_chain(k=5, temperature=0.3):
    """Return a RetrievalQA chain with tuned retriever and LLM."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain
