import os
import json
import time
import numpy as np
import streamlit as st
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import logging
from model_evaluation import evaluate_rag_response
from langchain.prompts import ChatPromptTemplate

# Set up logging
logging.basicConfig(filename='../logs/app.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# -------------- PAGE CONFIG ------------------
st.set_page_config(
    page_title="Clinical Diagnosis RAG 💊",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------- CUSTOM ADVANCED STYLING (Bootstrap-inspired) ------------------
st.markdown("""
<style>
/* Base styling */
body {
    background: linear-gradient(135deg, #0a0a0a, #1a1a1a);
    color: #ffffff;
    font-family: 'Roboto', sans-serif;
    overflow-x: hidden;
}

/* Headers */
h1, h2, h3, h4 {
    color: #ffffff;
    text-shadow: 0 0 10px rgba(0, 230, 230, 0.5);
    font-family: 'Montserrat', sans-serif;
    transition: all 0.3s ease;
}

/* Chat and input styling */
.stChatMessage, .stTextInput, .stMarkdown, .stExpander {
    background: rgba(255, 255, 255, 0.05);
    border: 2px solid #00e6e6;
    border-radius: 15px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 5px 15px rgba(0, 230, 230, 0.2);
    animation: fadeInSlide 0.5s ease-out;
}

@keyframes fadeInSlide {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Buttons (Bootstrap-like) */
.stButton > button {
    background: #00e6e6;
    color: #ffffff;
    border: none;
    border-radius: 25px;
    padding: 10px 25px;
    font-weight: bold;
    font-size: 16px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 230, 230, 0.4);
}

.stButton > button:hover {
    background: #ffffff;
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0, 230, 230, 0.6);
}

/* Metrics and expanders */
.metric-label, .metric-value {
    color: #ffffff !important;
}

.stExpander {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 15px;
    transition: all 0.3s ease;
}

.stExpander:hover {
    background: rgba(255, 255, 255, 0.07);
    box-shadow: 0 5px 15px rgba(0, 230, 230, 0.3);
}

/* Sidebar */
.sidebar .block-container {
    background: rgba(255, 255, 255, 0.02);
    border-right: 2px solid #00e6e6;
    padding: 20px;
    border-radius: 0 15px 15px 0;
    animation: slideInLeft 0.5s ease-out;
}

@keyframes slideInLeft {
    from {
        transform: translateX(-20px);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* Error and success messages */
.stSuccess, .stError {
    border-radius: 15px;
    padding: 15px;
    margin: 10px 0;
    animation: bounceIn 0.5s ease-out;
}

@keyframes bounceIn {
    from, 20%, 40%, 60%, 80%, to {
        animation-timing-function: cubic-bezier(0.215, 0.610, 0.355, 1.000);
    }
    0% {
        opacity: 0;
        transform: scale3d(.3, .3, .3);
    }
    20% {
        transform: scale3d(1.1, 1.1, 1.1);
    }
    40% {
        transform: scale3d(.9, .9, .9);
    }
    60% {
        opacity: 1;
        transform: scale3d(1.03, 1.03, 1.03);
    }
    80% {
        transform: scale3d(.97, .97, .97);
    }
    to {
        opacity: 1;
        transform: scale3d(1, 1, 1);
    }
}
</style>
""", unsafe_allow_html=True)

# -------------- LOTTIE ANIMATION (Enhanced) ---------------
def load_lottie(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load Lottie animation: {str(e)}")
        return None

lottie_path = "../assets/clinic-animation.json"
with st.sidebar:
    st.markdown("<h3 style='color: #00e6e6;'>Clinical Insights</h3>", unsafe_allow_html=True)
    if Path(lottie_path).exists():
        lottie_json = load_lottie(lottie_path)
        if lottie_json:
            st_lottie(lottie_json, height=250, key="sidebar-lottie", speed=1, loop=True, quality="low")

# -------------- API AND EMBEDDINGS INIT (Error-Free) ------------------
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file. Please check your configuration.")
    st.stop()

try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.0)
    embeddings = HuggingFaceEmbeddings(
        model_name="distilbert-base-uncased",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
except Exception as e:
    st.error(f"Failed to initialize LLM or embeddings: {str(e)}")
    st.stop()

st.title("🩺 Advanced Clinical Diagnosis Assistant")

# Load FAISS index
if "vectors" not in st.session_state:
    st.session_state.vectors = FAISS.load_local("../models/faiss_index", embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS index loaded into session state")

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------- CHAT INTERFACE (Advanced) ------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"<div class='message'>{message['content']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("💬 Ask a clinical question...", key="chat_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"<div class='message'>{prompt}</div>", unsafe_allow_html=True)

    if not st.session_state.vectors:
        with st.chat_message("assistant"):
            st.error("System initialization failed. Please check logs or data.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing clinical data..."):
                try:
                    prompt_template = ChatPromptTemplate.from_template("""
                    Answer based on these clinical resources: 
                    <context> 
                    {context} 
                    </context> 
                    Question: {input} 
                    Provide a concise, professional, and accurate response.
                    """)

                    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})
                    chain = create_retrieval_chain(
                        retriever,
                        create_stuff_documents_chain(llm, prompt_template)
                    )

                    response = chain.invoke({"input": prompt})
                    st.markdown(f"<div class='message'>{response['answer']}</div>", unsafe_allow_html=True)

                    eval_scores = evaluate_rag_response(response, embeddings)

                    with st.expander("📚 Reference Materials", expanded=False):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"""
                            <div class='reference'>
                                <strong>Source {i+1}:</strong> {doc.metadata['source']}<br>
                                <small>Excerpt: {doc.page_content[:200]}...</small>
                            </div>
                            """, unsafe_allow_html=True)

                    with st.expander("🧪 Evaluation Metrics", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Hit Rate", f"{eval_scores['hit_rate']:.2f}", help="Measures retrieval accuracy")
                        with col2:
                            st.metric("Faithfulness", f"{eval_scores['faithfulness']:.2f}", help="Measures response accuracy")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Error in response generation: {str(e)}")

            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

# -------------- SIDEBAR INFO ------------------
with st.sidebar:
    st.markdown("""
    <h4 style='color: #00e6e6;'>About</h4>
    <p>This AI-powered tool assists with clinical diagnosis using Retrieval-Augmented Generation. All documents from <code>Diagnosis_flowchart/</code> and <code>Finished/</code> are processed.</p>
    """, unsafe_allow_html=True)