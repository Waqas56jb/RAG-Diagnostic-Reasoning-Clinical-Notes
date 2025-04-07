import os
import json
import glob
import logging
from pathlib import Path
from langchain.docstore.document import Document
import time

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/data_ingestion.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_clinical_data():
    docs = []
    start_time = time.time()

    # Set project root as current directory
    project_root = os.path.abspath(os.path.dirname(__file__))
    flowchart_path = os.path.join(project_root, "Diagnosis_flowchart")
    finished_path = os.path.join(project_root, "Finished")

    # Load Diagnostic Flowcharts
    flowchart_files = glob.glob(os.path.join(flowchart_path, "*.json"))
    logger.info(f"Found {len(flowchart_files)} flowchart files.")

    for fpath in flowchart_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                content = f"""
DIAGNOSTIC FLOWCHART: {Path(fpath).stem}
Diagnostic Path: {data.get('diagnostic', 'No diagnostic path')}
Key Criteria: {data.get('knowledge', 'No criteria')}
"""
                docs.append(Document(
                    page_content=content.strip(),
                    metadata={"source": fpath, "type": "flowchart"}
                ))
        except Exception as e:
            logger.warning(f"Error loading flowchart {fpath}: {str(e)}")

    # Load Patient Cases
    case_dirs = glob.glob(os.path.join(finished_path, "*"))
    logger.info(f"Found {len(case_dirs)} case categories.")

    for category_dir in case_dirs:
        if os.path.isdir(category_dir):
            case_files = glob.glob(os.path.join(category_dir, "*.json"))
            logger.info(f"Found {len(case_files)} cases in category '{Path(category_dir).name}'.")

            for case_file in case_files:
                try:
                    with open(case_file, 'r', encoding='utf-8') as f:
                        case_data = json.load(f)
                        notes = "\n".join(f"{k}: {v}" for k, v in case_data.items() if k.startswith("input"))
                        content = f"""
PATIENT CASE: {Path(case_file).stem}
Category: {Path(category_dir).name}
Notes:
{notes}
""".strip()

                        docs.append(Document(
                            page_content=content,
                            metadata={"source": case_file, "type": "patient_case"}
                        ))
                except Exception as e:
                    logger.warning(f"Error loading case {case_file}: {str(e)}")

    if not docs:
        logger.error("No documents loaded. Check your folders.")
        raise ValueError("No documents found in Diagnosis_flowchart or Finished directories.")

    logger.info(f"Successfully loaded {len(docs)} documents in {time.time() - start_time:.2f} seconds.")
    return docs

def save_to_raw(docs):
    raw_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    output_path = os.path.join(raw_dir, "clinical_data.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs],
            f,
            indent=4,
            ensure_ascii=False
        )

    logger.info(f"Saved {len(docs)} documents to {output_path}")
    print(f"[INFO] Data saved to {output_path}")

if __name__ == "__main__":
    try:
        documents = load_clinical_data()
        save_to_raw(documents)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"[ERROR] {e}")
