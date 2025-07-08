import os
import streamlit as st
from dotenv import load_dotenv
from rag_chain import get_qa_chain

# --- 🔐 Load API Key (Streamlit Cloud or Local .env) ---
load_dotenv()
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY not found in st.secrets or .env")

os.environ["OPENAI_API_KEY"] = openai_key

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="FinSolve 💸 | Smart Loan Advisor",
    page_icon="💸",
    layout="centered"
)

# --- Custom Dark Theme Styling ---
st.markdown("""
<style>
    body, .main, .block-container {
        background-color: #1e272e;
        font-family: 'Segoe UI', sans-serif;
        color: #dcdde1;
    }
    .title {
        font-size: 2.8em;
        font-weight: 800;
        color: #00cec9;
        text-align: center;
        margin-bottom: 0;
    }
    .tagline {
        font-size: 1.05em;
        color: #74b9ff;
        text-align: center;
        margin-bottom: 40px;
    }
    .stTextInput > div > input {
        background-color: #2d3436 !important;
        color: #ffffff !important;
        border: 1px solid #636e72 !important;
        border-radius: 8px;
        padding: 0.8em;
    }
    .stButton > button {
        background-color: #00cec9;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #01a3a4;
        transform: scale(1.02);
    }
    .answer-box {
        background: linear-gradient(145deg, #2d3436, #1f2a2e);
        border: 1px solid #00cec9;
        border-left: 5px solid #00cec9;
        padding: 1.2em;
        border-radius: 12px;
        margin-top: 20px;
        font-size: 1.05em;
        color: #dfe6e9;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .footer {
        text-align: center;
        margin-top: 60px;
        font-size: 0.85em;
        color: #636e72;
        border-top: 1px solid #2d3436;
        padding-top: 15px;
    }
    .stExpander {
        background-color: #2f3640 !important;
        border-radius: 8px;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<div class='title'>FinSolve 💸</div>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>Hyderabad’s Premium Loan & EMI Advisor</div>", unsafe_allow_html=True)

# --- Input ---
query = st.text_input("💬 Ask your loan/EMI question", placeholder="E.g. Best EMI for 10 lakh in SBI bank?")

# --- Load QA Chain ---
@st.cache_resource(show_spinner=False)
def load_chain():
    return get_qa_chain()

qa_chain = load_chain()

# --- Button Action ---
if st.button("🔍 Get Answer"):
    if not query.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🧠 Thinking like a financial expert..."):
            try:
                result = qa_chain.invoke({"question": query})  # ✅ updated key from "query" to "question"
                answer = result.get("result", "❌ Sorry, couldn’t find an answer.")
                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

                if result.get("source_documents"):
                    with st.expander("📄 Sources"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.write(doc.page_content)

            except Exception as e:
                st.error("❌ Something went wrong.")
                with st.expander("🔧 Show error details"):
                    st.exception(e)

# --- Footer ---
st.markdown("<div class='footer'>Made by Vishal | Powered by LangChain & OpenAI</div>", unsafe_allow_html=True)
