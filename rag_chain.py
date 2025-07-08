import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from vector_store import build_vectorstore  # Optional auto rebuild

# ✅ Load API key from .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables.")

# ✅ Load or rebuild FAISS vectorstore
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_key
    )
    try:
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        print("⚠️ FAISS index missing or corrupted. Rebuilding from CSV...")
        build_vectorstore()
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# ✅ Build QA Chain with correct prompt and input key "question"
def get_qa_chain(k=15, temperature=0.3):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
        openai_api_key=openai_key
    )

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
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        },
        input_key="question"  # ✅ Must match Streamlit's input key
    )
