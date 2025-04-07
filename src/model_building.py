import json
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Set up logging
logging.basicConfig(filename='logs/model_building.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data():
    with open("../data/processed/processed_data.json", "r", encoding="utf-8") as f:
        processed_data = json.load(f)
    return [Document(page_content=item["content"], metadata=item["metadata"]) for item in processed_data]

def build_faiss_index(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="distilbert-base-uncased",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("../models/faiss_index")
    logger.info("FAISS index built and saved successfully")
    return vector_store

if __name__ == "__main__":
    processed_docs = load_processed_data()
    build_faiss_index(processed_docs)