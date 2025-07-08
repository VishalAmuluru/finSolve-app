import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ✅ Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Please check your .env or Streamlit secrets.")

def load_dataset():
    """Load and clean loan offers dataset from CSV."""
    try:
        df = pd.read_csv("data/hyderabad_loan_offers_10k.csv")
    except FileNotFoundError:
        raise FileNotFoundError("❌ CSV file not found. Check the path: data/hyderabad_loan_offers_10k.csv")

    df.columns = df.columns.str.strip().str.lower()
    df.fillna("Not specified", inplace=True)

    # ✅ Format text for embeddings
    formatted_texts = df.apply(
        lambda row: (
            f"Bank: {row['bank_name']} | Loan Type: {row['loan_type']} | Location: {row['location']}\n"
            f"Interest Rate: {row['interest_rate']}% | Tenure: {row['tenure_years']} years\n"
            f"Loan Amount: ₹{row['min_amount']} - ₹{row['max_amount']} | Processing Fee: ₹{row['processing_fee']}\n"
            f"Employment: {row['employment_type']} | Description: {row['description']}"
        ),
        axis=1
    )

    return [Document(page_content=text) for text in formatted_texts]

def build_vectorstore():
    """Build and save FAISS index with OpenAI embeddings."""
    print("🔄 Building FAISS index from dataset...")
    docs = load_dataset()

    try:
        # ✅ Use cost-effective model
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key
        )
    except Exception as e:
        raise RuntimeError("❌ Failed to initialize OpenAI embeddings.") from e

    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index")
        print("✅ FAISS index saved to 'faiss_index/'")
    except Exception as e:
        raise RuntimeError("❌ Failed to create or save FAISS index.") from e

if __name__ == "__main__":
    build_vectorstore()
