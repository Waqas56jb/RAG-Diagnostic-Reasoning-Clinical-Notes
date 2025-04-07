# src/model_building.py
import json
import logging
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Setup directories
BASE_DIR = Path(__file__).resolve().parent.parent
for folder in ["data/processed", "models/faiss_index", "logs"]:
    (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(filename=str(BASE_DIR / "logs" / "model_building.log"), level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data():
    processed_path = BASE_DIR / "data" / "processed" / "processed_data.json"
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found at {processed_path}")
    with open(processed_path, "r", encoding="utf-8") as f:
        processed_data = json.load(f)
    return [Document(page_content=item["content"], metadata=item["metadata"]) for item in processed_data]

def build_faiss_index(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="distilbert-base-uncased",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.from_documents(documents, embeddings)
    faiss_path = BASE_DIR / "models" / "faiss_index"
    faiss_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(faiss_path))
    logger.info("FAISS index built and saved successfully")
    return vector_store

if __name__ == "__main__":
    processed_docs = load_processed_data()
    build_faiss_index(processed_docs)