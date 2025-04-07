import json
import logging
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Set up logging
logging.basicConfig(filename='logs/feature_engineering.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def load_interim_data():
    with open("../data/interim/splits.json", "r", encoding="utf-8") as f:
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
    os.makedirs("../data/processed", exist_ok=True)
    with open("../data/processed/processed_data.json", "w", encoding="utf-8") as f:
        json.dump([{"content": doc.page_content, "metadata": doc.metadata} for doc in documents], f, indent=4)

if __name__ == "__main__":
    interim_docs = load_interim_data()
    processed_docs, embeddings = generate_embeddings(interim_docs)
    save_to_processed(processed_docs)