# src/app.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import logging
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from streamlit_lottie import st_lottie

from data_preprocessing import init_knowledge_base
from model_evaluation import evaluate_rag_response
from data_ingestion import load_clinical_data
from utils.load_params import load_params

# Load params
params = load_params()
llm_cfg = params["llm"]

# Setup directories
BASE_DIR = Path(__file__).resolve().parent.parent
for folder in ["data/raw", "data/interim", "data/processed", "Diagnosis_flowchart", "Finished", "models/faiss_index", "logs", "reports", "assets"]:
    (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)

# Logging
log_dir = BASE_DIR / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("GROQ_API_KEY missing in .env")
    logger.error("Missing API key")
    st.stop()

# Page config & CSS with Animated Background (Blue, White, and Black Theme)
st.set_page_config(page_title="Clinical Diagnosis RAG 💊", page_icon="🩺", layout="wide")

# HTML and CSS for Animated Background and Modern Blue-White-Black UI
st.markdown("""
<!DOCTYPE html>
<html>
<head>
    <style>
        body, .stApp {
            margin: 0;
            padding: 0;
            height: 100vh;
            overflow: hidden;
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(135deg, #000000, #1e88e5, #ffffff);
        }
        .animated-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, #1e88e5 0%, #ffffff 50%, #000000 100%);
            background-size: 200% 200%;
            animation: gradientShift 15s ease infinite;
            z-index: -1;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 0%; }
            50% { background-position: 100% 100%; }
            100% { background-position: 0% 0%; }
        }
        .content {
            position: relative;
            z-index: 1;
            padding: 40px;
            max-width: 900px;
            margin: 30px auto;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 25px;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
            border: 3px solid #1e88e5;
            color: #333333;
            overflow: hidden;
        }
        .stTitle {
            color: #1e88e5;
            text-align: center;
            font-size: 3.2em;
            font-weight: 700;
            text-shadow: 2px 2px 6px rgba(30, 136, 229, 0.3);
            margin-bottom: 25px;
            animation: fadeInZoom 1.5s ease-out;
        }
        @keyframes fadeInZoom {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }
        .stChatMessage {
            background: #ffffff;
            border-radius: 18px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            border-left: 6px solid #1e88e5;
            animation: slideInUp 0.8s ease-out;
        }
        @keyframes slideInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stButton>button {
            background: linear-gradient(45deg, #1e88e5, #1976d2);
            color: #ffffff;
            border-radius: 30px;
            padding: 15px 30px;
            border: none;
            transition: all 0.4s ease;
            font-weight: 600;
            box-shadow: 0 6px 20px rgba(30, 136, 229, 0.4);
            position: relative;
            overflow: hidden;
        }
        .stButton>button:before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(255, 255, 255, 0.4);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: all 0.6s ease;
            z-index: 0;
        }
        .stButton>button:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(25, 118, 210, 0.6);
        }
        .stButton>button:hover:before {
            width: 400px;
            height: 400px;
        }
        .stExpander {
            background: #ffffff;
            border-radius: 18px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            border-left: 6px solid #1e88e5;
            margin: 15px 0;
            animation: expandFade 0.7s ease-out;
        }
        @keyframes expandFade {
            from { opacity: 0; height: 0; }
            to { opacity: 1; height: auto; }
        }
        .stExpander > div > div {
            padding: 20px;
            color: #333333;
        }
        .sidebar .block-container {
            display: none; /* Hide sidebar content as requested */
            background: linear-gradient(135deg, #000000, #2c2c2c);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            animation: fadeIn 1.2s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .metric-container {
            background: linear-gradient(135deg, #2c2c2c, #424242);
            border-radius: 18px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            border-left: 6px solid #1e88e5;
            animation: popUp 0.9s ease-out;
        }
        @keyframes popUp {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
        .stSpinner {
            color: #1e88e5;
            font-size: 1.5em;
            text-align: center;
            animation: bouncePulse 1.5s infinite;
        }
        @keyframes bouncePulse {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-15px); box-shadow: 0 5px 15px rgba(30, 136, 229, 0.6); }
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.11.4/dist/gsap.min.js"></script>
    <script>
        gsap.from(".stTitle", { duration: 1.5, scale: 0.9, opacity: 0, ease: "elastic.out(1, 0.3)" });
        gsap.from(".stChatMessage", { duration: 0.7, y: 20, opacity: 0, stagger: 0.3, ease: "power2.out" });
        gsap.from(".stExpander", { duration: 0.8, y: 30, opacity: 0, stagger: 0.2, ease: "back.out(1.7)" });
    </script>
</head>
<body>
    <div class="animated-background"></div>
    <div class="content">
""", unsafe_allow_html=True)

# Sidebar Lottie (Hidden content as per request)
def load_lottie(fp):
    try:
        return json.loads(Path(fp).read_text())
    except Exception as e:
        logger.warning(f"Lottie load failed: {e}")
        return None

# Sidebar is present but content is hidden
with st.sidebar:
    st.markdown("<style>.sidebar .block-container { display: none !important; }</style>", unsafe_allow_html=True)
    lottie_fp = BASE_DIR / "assets" / "clinic-animation.json"
    anim = load_lottie(str(lottie_fp))
    if anim:
        st_lottie(anim, height=220, loop=True, quality="high")

# Initialize LLM & embeddings
logger.info("Initializing LLM and embeddings")
start = time.time()
try:
    llm = ChatGroq(groq_api_key=groq_key,
                   model_name=llm_cfg["model_name"],
                   temperature=llm_cfg["temperature"])
    embeddings = HuggingFaceEmbeddings(
        model_name=params["embedding"]["model_name"],
        model_kwargs={"device": params["embedding"]["device"]},
        encode_kwargs={"normalize_embeddings": params["embedding"]["normalize_embeddings"]}
    )
except Exception as e:
    st.error(f"Init failed: {e}")
    logger.error(e)
    st.stop()
logger.info(f"LLM ready in {time.time() - start:.2f}s")

st.markdown("<h1 class='stTitle'>🩺 Advanced Clinical Diagnosis Assistant</h1>", unsafe_allow_html=True)

# Load or build FAISS index
if "vectors" not in st.session_state:
    idx_path = BASE_DIR / "models" / "faiss_index"
    if not (idx_path / "index.faiss").exists():
        st.info("Building FAISS index, please wait…")
        st.session_state.vectors = init_knowledge_base()
    else:
        st.session_state.vectors = FAISS.load_local(str(idx_path), embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS index loaded")

if not st.session_state.vectors:
    st.error("Knowledge base initialization failed.")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='stChatMessage'>{msg['content']}</div>", unsafe_allow_html=True)

# User input
if user_q := st.chat_input("💬 Ask a clinical question…"):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("assistant"):
        with st.spinner("Analyzing clinical data…"):
            try:
                tmpl = ChatPromptTemplate.from_template("""
Based on these clinical resources, provide:
**Disease**, **Diagnostic Features**, **Potential Treatments**.
<context>
{context}
</context>
Question: {input}
""")
                retriever = st.session_state.vectors.as_retriever(
                    search_kwargs={"k": params["retrieval"]["top_k"]}
                )
                chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, tmpl))
                response = chain.invoke({"input": user_q})
                st.markdown(f"<div class='stChatMessage'>{response['answer']}</div>", unsafe_allow_html=True)

                scores = evaluate_rag_response(response, embeddings, retriever)

                with st.expander("📚 Reference Materials", expanded=False):
                    for i, d in enumerate(response.get("context", []), 1):
                        st.markdown(f"<h4 style='color:#1e88e5;'>Source {i}:</h4> {Path(d.metadata['source']).name}", unsafe_allow_html=True)
                        st.code(d.page_content, language="text")

                with st.expander("🧪 Evaluation Metrics", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"<div class='metric-container'><h4>Hit Rate</h4><p style='color:#ffffff; font-size:1.2em;'>%.2f</p></div>" % scores['hit_rate'], unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div class='metric-container'><h4>Faithfulness</h4><p style='color:#ffffff; font-size:1.2em;'>%.2f</p></div>" % scores['faithfulness'], unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
                logger.error(e)

    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

# Sidebar About (Hidden as per request)
st.markdown("<style>.sidebar .block-container { display: none !important; }</style>", unsafe_allow_html=True)

st.markdown("""
    </div>
</body>
</html>
""", unsafe_allow_html=True)