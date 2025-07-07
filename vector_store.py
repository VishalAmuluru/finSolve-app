import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

def load_dataset():
    try:
        df = pd.read_csv("data/hyderabad_loan_offers_10k.csv")
    except FileNotFoundError:
        raise Exception("CSV file not found. Check the path to your data file.")
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Fill missing values to avoid formatting issues
    df = df.fillna("Not specified")

    # Format data into structured descriptive strings
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
    print("Building FAISS index...")
    docs = load_dataset()

    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise Exception("Error initializing OpenAI embeddings. Check your API key.") from e

    # Create FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save index to disk
    vectorstore.save_local("faiss_index")
    print("FAISS index saved to 'faiss_index/' directory.")

if __name__ == "__main__":
    build_vectorstore()
