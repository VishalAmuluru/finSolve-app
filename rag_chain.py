import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ✅ Set API Key from Streamlit Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def load_vectorstore():
    """Build FAISS index from local data.txt file (normalized)."""
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            text = f.read().lower()  # Normalize bank names
    except FileNotFoundError:
        st.error("❌ data.txt file not found. Please upload or generate it.")
        st.stop()

    # Split and chunk text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    # ✅ Specify model to avoid OpenAI BadRequestError
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    return FAISS.from_documents(docs, embeddings)

def get_qa_chain(k=15, temperature=0.3):
    """Create a RetrievalQA chain with tuned retriever and OpenAI LLM."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # ✅ Using newer, preferred way
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)

    prompt = PromptTemplate.from_template(
        """You are a smart and unbiased loan advisor. Provide information using all relevant bank offers,
not just well-known ones. Be clear, concise, and useful.

Question: {query}"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
