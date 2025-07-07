import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ✅ Set the OpenAI key securely from Streamlit Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def load_vectorstore():
    """Load the FAISS index with OpenAI embeddings."""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

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
