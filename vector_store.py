import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

# ✅ Load .env variables (for local dev)
load_dotenv()

# ✅ Try loading from Streamlit secrets if available
try:
    import streamlit as st
    openai_key = st.secrets.get("OPENAI_API_KEY", None)
except:
    st = None
    openai_key = None

# ✅ Fallback to .env if not found in Streamlit
if not openai_key:
    openai_key = os.getenv("OPENAI_API_KEY")

# ✅ Raise error if still missing
if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env or Streamlit secrets.")

# ✅ Load and format dataset
def load_dataset():
    try:
        df = pd.read_csv("data/hyderabad_loan_offers_10k.csv")
    except FileNotFoundError:
        raise FileNotFoundError("❌ CSV not found: data/hyderabad_loan_offers_10k.csv")

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

# ✅ Build FAISS index
def build_vectorstore():
    print("🔄 Loading and chunking data...")
    raw_texts = load_dataset()

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for text in raw_texts:
        chunks.extend(splitter.split_text(text))

    documents = [Document(page_content=chunk) for chunk in chunks]
    print(f"✅ {len(documents)} chunks ready for embedding.")

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
        print("✅ FAISS index saved to 'faiss_index/'")
    except Exception as e:
        raise RuntimeError("❌ Failed to build or save FAISS index.") from e

if __name__ == "__main__":
    build_vectorstore()
