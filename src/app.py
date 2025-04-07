# src/app.py
import sys, os
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
    filename=log_dir/"app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("GROQ_API_KEY missing in .env"); logger.error("Missing API key"); st.stop()

# Page config & CSS with Particle Background (State-of-the-Art UI)
st.set_page_config(page_title="Clinical Diagnosis RAG 💊", page_icon="🩺", layout="wide")

# HTML and CSS for Particle Background and Modern UI
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
            font-family: 'Arial', sans-serif;
            background: #f0f8ff;
        }
        .particle-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }
        .content {
            position: relative;
            z-index: 1;
            padding: 20px;
            background: rgba(240, 248, 255, 0.9);
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        .stTitle {
            color: #1a5276;
            text-align: center;
            font-size: 3.5em;
            font-weight: bold;
            text-shadow: 2px 2px 6px #888888;
            margin-bottom: 30px;
            animation: fadeIn 1.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stChatMessage {
            background: #ffffff;
            border-radius: 20px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            border-left: 6px solid #00e6e6;
            animation: slideIn 0.5s ease-out;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .stButton>button {
            background: linear-gradient(45deg, #00e6e6, #00b3b3);
            color: white;
            border-radius: 30px;
            padding: 15px 30px;
            border: none;
            transition: all 0.3s ease;
            font-weight: bold;
            box-shadow: 0 6px 15px rgba(0, 230, 230, 0.3);
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
            background: rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: all 0.6s ease;
            z-index: 0;
        }
        .stButton>button:hover {
            transform: scale(1.1);
            box-shadow: 0 8px 20px rgba(0, 230, 230, 0.5);
        }
        .stButton>button:hover:before {
            width: 300px;
            height: 300px;
        }
        .stExpander {
            background: #ffffff;
            border-radius: 20px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            border-left: 6px solid #00e6e6;
            margin: 10px 0;
            animation: fadeInUp 0.8s ease-out;
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stExpander > div > div {
            padding: 20px;
        }
        .sidebar .block-container {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
            animation: slideInLeft 1s ease-out;
        }
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .metric-container {
            background: linear-gradient(135deg, #e9ecef, #dee2e6);
            border-radius: 20px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
            border-left: 6px solid #00e6e6;
            animation: bounceIn 0.8s ease-out;
        }
        @keyframes bounceIn {
            from { opacity: 0; transform: scale(0.8); }
            to { opacity: 1; transform: scale(1); }
        }
        .stSpinner {
            color: #00e6e6;
            font-size: 1.5em;
            text-align: center;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>
    <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
    <script>
        particlesJS("particles-js", {
            "particles": {
                "number": {"value": 80, "density": {"enable": true, "value_area": 800}},
                "color": {"value": "#00e6e6"},
                "shape": {"type": "circle"},
                "opacity": {"value": 0.5, "random": true},
                "size": {"value": 3, "random": true},
                "line_linked": {"enable": true, "distance": 150, "color": "#00e6e6", "opacity": 0.4, "width": 1},
                "move": {"enable": true, "speed": 6, "direction": "none", "random": false, "straight": false, "out_mode": "out", "bounce": false}
            },
            "interactivity": {
                "detect_on": "canvas",
                "events": {"onhover": {"enable": true, "mode": "repulse"}, "onclick": {"enable": true, "mode": "push"}},
                "modes": {"repulse": {"distance": 100, "duration": 0.4}, "push": {"particles_nb": 4}}
            },
            "retina_detect": true
        });
    </script>
</head>
<body>
    <div id="particles-js" class="particle-background"></div>
    <div class="content">
""", unsafe_allow_html=True)

# Sidebar Lottie
def load_lottie(fp):
    try: return json.loads(Path(fp).read_text())
    except Exception as e:
        logger.warning(f"Lottie load failed: {e}")
        return None

with st.sidebar:
    st.markdown("<h3 style='color:#00e6e6; text-shadow: 1px 1px 3px #888888;'>Clinical Insights</h3>", unsafe_allow_html=True)
    lottie_fp = BASE_DIR / "assets" / "clinic-animation.json"
    anim = load_lottie(str(lottie_fp))
    if anim: st_lottie(anim, height=250, loop=True, quality="high")

# Initialize LLM & embeddings
logger.info("Initializing LLM and embeddings")
start = time.time()
try:
    llm = ChatGroq(groq_api_key=groq_key,
                  model_name=llm_cfg["model_name"],
                  temperature=llm_cfg["temperature"])
    embeddings = HuggingFaceEmbeddings(
        model_name=params["embedding"]["model_name"],
        model_kwargs={"device":params["embedding"]["device"]},
        encode_kwargs={"normalize_embeddings":params["embedding"]["normalize_embeddings"]}
    )
except Exception as e:
    st.error(f"Init failed: {e}"); logger.error(e); st.stop()
logger.info(f"LLM ready in {time.time()-start:.2f}s")

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
    st.error("Knowledge base initialization failed."); st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='stChatMessage'>{msg['content']}</div>", unsafe_allow_html=True)

# User input
if user_q := st.chat_input("💬 Ask a clinical question…"):
    st.session_state.messages.append({"role":"user","content":user_q})
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
                    search_kwargs={"k":params["retrieval"]["top_k"]}
                )
                chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, tmpl))
                response = chain.invoke({"input":user_q})
                st.markdown(f"<div class='stChatMessage'>{response['answer']}</div>", unsafe_allow_html=True)

                scores = evaluate_rag_response(response, embeddings, retriever)

                with st.expander("📚 Reference Materials", expanded=False):
                    for i,d in enumerate(response.get("context",[]),1):
                        st.markdown(f"<h4 style='color:#1a5276;'>Source {i}:</h4> {Path(d.metadata['source']).name}", unsafe_allow_html=True)
                        st.code(d.page_content, language="text")

                with st.expander("🧪 Evaluation Metrics", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("<div class='metric-container'><h4>Hit Rate</h4><p style='color:#00e6e6; font-size:1.2em;'>%.2f</p></div>" % scores['hit_rate'], unsafe_allow_html=True)
                    with c2:
                        st.markdown("<div class='metric-container'><h4>Faithfulness</h4><p style='color:#00e6e6; font-size:1.2em;'>%.2f</p></div>" % scores['faithfulness'], unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}"); logger.error(e)

    st.session_state.messages.append({"role":"assistant","content":response["answer"]})

# Sidebar About
with st.sidebar:
    st.markdown("""
<h4 style='color:#00e6e6; text-shadow: 1px 1px 3px #888888;'>About</h4>
<p style='color:#2c3e50;'>This AI tool uses RAG to analyze clinical JSON files from <code>Diagnosis_flowchart/</code> and <code>Finished/</code>.</p>
""", unsafe_allow_html=True)

st.markdown("""
    </div>
</body>
</html>
""", unsafe_allow_html=True)