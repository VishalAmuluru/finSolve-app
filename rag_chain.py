import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document  # Required to build docs manually

# ✅ Set OpenAI API Key securely from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def load_vectorstore():
    """Build FAISS index from local data.txt file (normalized)."""
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            text = f.read().lower()  # Normalize bank names
    except FileNotFoundError:
        st.error("❌ data.txt file not found. Please upload or generate it.")
        st.stop()

    # ✅ Split large text into safe chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # ✅ Convert each chunk into a Document object
    documents = [Document(page_content=chunk) for chunk in chunks]

    # ✅ Use token-safe model for embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # ✅ Build FAISS index
    return FAISS.from_documents(documents, embeddings)

def get_qa_chain(k=15, temperature=0.3):
    """Create a RetrievalQA chain with a tuned retriever and prompt."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)

    # Prompt that encourages diversity in answers
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
