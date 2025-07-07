import os
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA

# Load environment variables from .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Please check your .env file.")

def load_vectorstore():
    """Load the FAISS index using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_qa_chain(k=5, temperature=0.3):
    """Return a RetrievalQA chain with retriever + LLM."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature, openai_api_key=openai_key)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain

# CLI Testing
if __name__ == "__main__":
    qa_chain = get_qa_chain()
    print("Ask a question (type 'exit' to quit):")
    while True:
        query = input("\nYou: ")
        if query.strip().lower() == "exit":
            print("Exiting...")
            break

        response = qa_chain.invoke({"query": query})
        print("\nAnswer:", response["result"])

        # Optional: source document debugging
        if response.get("source_documents"):
            print("\nSources:")
            for i, doc in enumerate(response["source_documents"], start=1):
                print(f"[{i}] {doc.metadata.get('source', 'Document')} — {doc.page_content[:120]}...\n")
