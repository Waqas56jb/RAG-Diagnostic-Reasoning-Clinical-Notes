# src/data_preprocessing.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import logging
import time
from pathlib import Path

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from data_ingestion import load_clinical_data
from utils.load_params import load_params

# Load params
params = load_params()
chunk_size = params["text_splitter"]["chunk_size"]
chunk_overlap = params["text_splitter"]["chunk_overlap"]
emb_cfg = params["embedding"]

# Setup directories
BASE_DIR = Path(__file__).resolve().parent.parent
for folder in ["data/raw", "data/interim", "data/processed", "models/faiss_index", "logs", "reports"]:
    (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)

# Logging setup
log_dir = BASE_DIR / "logs"
log_dir.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_dir / "data_preprocessing.log")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.handlers = [fh, ch]

@st.cache_resource(show_spinner=False)
def init_knowledge_base():
    logger.info("Starting knowledge base initialization")
    start_time = time.time()

    docs = load_clinical_data()
    if not docs:
        logger.error("No docs to preprocess")
        st.error("No documents for preprocessing.")
        return None

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(docs)
    logger.info(f"Split into {len(splits)} chunks")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_cfg["model_name"],
        model_kwargs={"device":emb_cfg["device"]},
        encode_kwargs={"normalize_embeddings":emb_cfg["normalize_embeddings"]}
    )

    # FAISS
    vector_store = FAISS.from_documents(splits, embeddings)
    faiss_path = BASE_DIR / "models" / "faiss_index"
    faiss_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(faiss_path))
    logger.info(f"Saved FAISS index to {faiss_path}")

    logger.info(f"Knowledge base ready in {time.time()-start_time:.2f}s")
    return vector_store