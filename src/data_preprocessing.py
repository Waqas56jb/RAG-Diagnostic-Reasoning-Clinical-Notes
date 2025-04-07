import json
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Set up logging
logging.basicConfig(filename='logs/data_preprocessing.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data():
    raw_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data/raw/clinical_data.json")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [Document(page_content=item["content"], metadata=item["metadata"]) for item in raw_data]

def preprocess_data(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = splitter.split_documents(documents)
    logger.info(f"Preprocessed {len(documents)} documents into {len(splits)} splits")
    return splits

def save_to_interim(splits):
    interim_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data/interim")
    os.makedirs(interim_dir, exist_ok=True)
    with open(os.path.join(interim_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump([{"content": split.page_content, "metadata": split.metadata} for split in splits], f, indent=4)

if __name__ == "__main__":
    raw_docs = load_raw_data()
    splits = preprocess_data(raw_docs)
    save_to_interim(splits)