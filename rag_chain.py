import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# ✅ Get the OpenAI key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def load_vectorstore():
    """Build FAISS index from data.txt"""
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    # Embed and store in FAISS
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

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
