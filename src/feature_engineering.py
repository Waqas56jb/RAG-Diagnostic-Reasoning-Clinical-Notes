# src/feature_engineering.py
import json
import logging
import os
from pathlib import Path
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Setup directories
BASE_DIR = Path(__file__).resolve().parent.parent
for folder in ["data/interim", "data/processed", "logs"]:
    (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(filename=str(BASE_DIR / "logs" / "feature_engineering.log"), level=logging.INFO)
logger = logging.getLogger(__name__)

def load_interim_data():
    interim_path = BASE_DIR / "data" / "interim" / "splits.json"
    if not interim_path.exists():
        raise FileNotFoundError(f"Interim data not found at {interim_path}")
    with open(interim_path, "r", encoding="utf-8") as f:
        interim_data = json.load(f)
    return [Document(page_content=item["content"], metadata=item["metadata"]) for item in interim_data]

def generate_embeddings(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="distilbert-base-uncased",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    logger.info("Embeddings generated successfully")
    return documents, embeddings

def save_to_processed(documents):
    processed_path = BASE_DIR / "data" / "processed" / "processed_data.json"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump([{"content": doc.page_content, "metadata": doc.metadata} for doc in documents], f, indent=4)

if __name__ == "__main__":
    interim_docs = load_interim_data()
    processed_docs, embeddings = generate_embeddings(interim_docs)
    save_to_processed(processed_docs)