import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

# ✅ Load API Key from secrets or .env
load_dotenv()

if "OPENAI_API_KEY" in st.secrets:
    openai_key = st.secrets["OPENAI_API_KEY"]
elif os.getenv("OPENAI_API_KEY"):
    openai_key = os.getenv("OPENAI_API_KEY")
else:
    raise ValueError("❌ OPENAI_API_KEY not found in secrets or .env")

# ✅ Load and clean CSV dataset
def load_dataset():
    try:
        df = pd.read_csv("data/hyderabad_loan_offers_10k.csv")
    except FileNotFoundError:
        raise FileNotFoundError("❌ CSV file not found: data/hyderabad_loan_offers_10k.csv")

    df.columns = df.columns.str.strip().str.lower()
    df.fillna("Not specified", inplace=True)

    formatted_texts = df.apply(
        lambda row: (
            f"Bank: {row['bank_name']} | Loan Type: {row['loan_type']} | Location: {row['location']}\n"
            f"Interest Rate: {row['interest_rate']}% | Tenure: {row['tenure_years']} years\n"
            f"Loan Amount: ₹{row['min_amount']} - ₹{row['max_amount']} | Processing Fee: ₹{row['processing_fee']}\n"
            f"Employment: {row['employment_type']} | Description: {row['description']}"
        ),
        axis=1
    ).tolist()

    return formatted_texts

# ✅ Build and save FAISS vectorstore
def build_vectorstore():
    print("🔄 Loading and chunking data...")
    raw_texts = load_dataset()

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for text in raw_texts:
        chunks.extend(splitter.split_text(text))

    documents = [Document(page_content=chunk) for chunk in chunks]
    print(f"✅ {len(documents)} text chunks ready for embedding.")

    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key
        )
    except Exception as e:
        raise RuntimeError("❌ Failed to initialize OpenAI embeddings.") from e

    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("faiss_index")
        print("✅ FAISS index successfully saved to 'faiss_index/'")
    except Exception as e:
        raise RuntimeError("❌ Failed to build or save FAISS index.") from e

if __name__ == "__main__":
    build_vectorstore()
