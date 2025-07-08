import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ✅ Load from .env (for local) OR st.secrets (for Streamlit Cloud)
load_dotenv()

openai_key = None
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY not found in st.secrets or .env")

# ✅ Load FAISS index from disk
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_key
    )
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# ✅ Define QA chain using proper prompt format
def get_qa_chain(k=15, temperature=0.3):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature, openai_api_key=openai_key)

    prompt = PromptTemplate.from_template(
        """You are a smart and unbiased loan advisor. Use the following context to answer the question.
Include all relevant bank offers, not just popular ones.

Context:
{context}

Question:
{question}

Answer:"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
        input_key="question"  # ✅ required to match the prompt input variable
    )
