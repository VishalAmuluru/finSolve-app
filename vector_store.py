import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # ✅ Use updated OpenAI import
from langchain_core.documents import Document

# ✅ Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Make sure it's set in your .env file.")

def load_dataset():
    try:
        df = pd.read_csv("data/hyderabad_loan_offers_10k.csv")
    except FileNotFoundError:
        raise FileNotFoundError("❌ CSV file not found. Check the path: 'data/hyderabad_loan_offers_10k.csv'")

    df.columns = df.columns.str.strip().str.lower()
    df = df.fillna("Not specified")

    combined = df.apply(
        lambda row: (
            f"{row['bank_name']} offers a {row['loan_type']} loan in {row['location']}.\n"
            f"Interest Rate: {row['interest_rate']}% | Tenure: {row['tenure_years']} years\n"
            f"Amount Range: ₹{row['min_amount']} - ₹{row['max_amount']}\n"
            f"Processing Fee: ₹{row['processing_fee']} | Employment Type: {row['employment_type']}\n"
            f"More Info: {row['description']}"
        ),
        axis=1
    )

    return [Document(page_content=text) for text in combined]

def build_vectorstore():
    print("🔄 Building FAISS index...")
    docs = load_dataset()

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)  # ✅ Pass key explicitly
    except Exception as e:
        raise Exception("❌ Failed to initialize OpenAI embeddings.") from e

    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index")
        print("✅ FAISS index saved to 'faiss_index/'")
    except Exception as e:
        raise Exception("❌ Failed to create or save FAISS vectorstore.") from e

if __name__ == "__main__":
    build_vectorstore()
