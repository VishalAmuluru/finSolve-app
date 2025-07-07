import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

def load_vectorstore():
    """Load the FAISS index with OpenAI embeddings."""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_qa_chain(k=5, temperature=0.3):
    """Return a RetrievalQA chain with tuned retriever and LLM."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain

# Optional CLI usage for testing
if __name__ == "__main__":
    qa_chain = get_qa_chain()

    print("Ask a question (type 'exit' to quit):")
    while True:
        query = input("\nYou: ")
        if query.lower().strip() == "exit":
            print("Exiting...")
            break

        response = qa_chain.invoke({"query": query})
        print("\nAnswer:", response["result"])

        # Optional: print sources for debugging
        if response.get("source_documents"):
            print("\nSources:")
            for i, doc in enumerate(response["source_documents"], start=1):
                print(f"[{i}] {doc.metadata.get('source', 'Document')} — {doc.page_content[:120]}...\n")
