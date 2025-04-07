# src/data_ingestion.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import logging
import time
from pathlib import Path

import streamlit as st
from langchain.docstore.document import Document
from utils.load_params import load_params

# Load params
params = load_params()
max_files = params["data_ingestion"]["max_files"]

# Setup directories
BASE_DIR = Path(__file__).resolve().parent.parent
for folder in ["data/raw", "data/interim", "data/processed", "Diagnosis_flowchart", "Finished", "models/faiss_index", "logs", "reports"]:
    (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)

# Logging setup
log_dir = BASE_DIR / "logs"
log_dir.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_dir / "data_ingestion.log")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.handlers = [fh, ch]

@st.cache_data(show_spinner=False)
def load_clinical_data():
    logger.info("Starting clinical data ingestion")
    start_time = time.time()
    docs = []

    # Flowchart files
    flow_dir = BASE_DIR / "Diagnosis_flowchart"
    if flow_dir.exists():
        files = list(flow_dir.glob("*.json"))[:max_files]
        logger.info(f"Found {len(files)} flowchart files")
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                content = (
                    f"DIAGNOSTIC FLOWCHART: {f.stem}\n"
                    f"Diagnostic Path: {data.get('diagnostic','No diagnostic path')}\n"
                    f"Key Criteria: {data.get('knowledge','No criteria')}"
                )
                docs.append(Document(page_content=content,
                                     metadata={"source":str(f),"type":"flowchart"}))
                logger.info(f"Loaded flowchart: {f.name}")
            except Exception as e:
                logger.warning(f"Failed to load {f.name}: {e}")
    else:
        logger.warning(f"Directory not found: {flow_dir}. Creating it.")
        flow_dir.mkdir(parents=True, exist_ok=True)

    # Patient case files
    remaining = max_files - len(docs)
    if remaining > 0:
        finished_dir = BASE_DIR / "Finished"
        if finished_dir.exists():
            all_cases = []
            for cat in finished_dir.iterdir():
                if cat.is_dir():
                    all_cases.extend(cat.glob("*.json"))
            for case in all_cases[:remaining]:
                try:
                    data = json.loads(case.read_text(encoding="utf-8"))
                    notes = "\n".join(f"{k}: {v}" for k,v in data.items() if k.startswith("input"))
                    content = (
                        f"PATIENT CASE: {case.stem}\n"
                        f"Category: {case.parent.name}\n"
                        f"Notes: {notes}"
                    )
                    docs.append(Document(page_content=content,
                                         metadata={"source":str(case),"type":"patient_case"}))
                    logger.info(f"Loaded patient case: {case.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {case.name}: {e}")
        else:
            logger.warning(f"Directory not found: {finished_dir}. Creating it.")
            finished_dir.mkdir(parents=True, exist_ok=True)

    if not docs:
        logger.error("No clinical data found")
        st.error("No clinical data found in Diagnosis_flowchart or Finished.")
        return []

    duration = time.time() - start_time
    logger.info(f"Ingested {len(docs)} docs in {duration:.2f}s")

    # Save for DVC
    out_path = BASE_DIR / "data" / "interim" / "splits.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([{"content":d.page_content,"metadata":d.metadata} for d in docs], f, indent=4)
    logger.info(f"Saved ingestion output to {out_path}")

    return docs