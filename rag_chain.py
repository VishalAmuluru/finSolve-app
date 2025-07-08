import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ✅ Load key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def load_vectorstore():
    """Load prebuilt FAISS index from disk (created using vector_store.py)."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_qa_chain(k=15, temperature=0.3):
    """Return RetrievalQA chain using FAISS and OpenAI."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)

    prompt = PromptTemplate.from_template(
        """You are a smart and unbiased loan advisor. Provide details using all relevant bank offers,
not just popular ones. Be clear, concise, and helpful.

Question: {query}"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
