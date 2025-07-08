import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

# 🔐 Load OpenAI API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY not found")

# 📄 Load and format loan dataset
def load_dataset():
    df = pd.read_csv("data/hyderabad_loan_offers_10k.csv")
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

# 🧠 Build and save FAISS vector index
def build_vectorstore():
    raw_texts = load_dataset()
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    chunks = []
    for text in raw_texts:
        chunks.extend(splitter.split_text(text))

    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ FAISS vector store saved to 'faiss_index/'")

# 🏁 Run if main
if __name__ == "__main__":
    build_vectorstore()
