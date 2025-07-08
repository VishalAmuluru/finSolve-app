import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ✅ Load API Key securely from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

def load_vectorstore():
    """🔍 Load the FAISS index previously built from CSV."""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.load_local(
            folder_path="faiss_index",
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error("❌ Failed to load FAISS index. Make sure 'faiss_index' folder exists.")
        raise e

def get_qa_chain(k=15, temperature=0.3):
    """🔗 Set up RetrievalQA chain using FAISS and OpenAI LLM."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Use cost-effective and reliable LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)

    # Prompt encourages diverse and inclusive answers
    prompt = PromptTemplate.from_template(
        """You are a helpful and unbiased loan advisor. Provide details using all relevant bank offers,
not just the most popular ones. Answer clearly, concisely, and accurately.

Question: {query}"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
