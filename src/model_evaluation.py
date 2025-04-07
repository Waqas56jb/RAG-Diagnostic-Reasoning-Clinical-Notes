import json
import logging
from sentence_transformers import util
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dvclive import Live

# Set up logging
logging.basicConfig(filename='logs/model_evaluation.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_hit_rate(retriever, query, expected_docs, k=1):
    try:
        retrieved_docs = retriever.get_relevant_documents(query, k=k)
        retrieved_contents = [doc.page_content for doc in retrieved_docs if hasattr(doc, 'page_content')]
        hits = 0
        for expected in expected_docs:
            if any(expected in retrieved for retrieved in retrieved_contents):
                hits += 1
        return hits / len(expected_docs) if expected_docs else 0.0
    except Exception as e:
        logger.error(f"Error in calculate_hit_rate: {str(e)}")
        return 0.0

def evaluate_rag_response(response, embeddings):
    scores = {}
    try:
        answer_embed = embeddings.embed_query(response.get("answer", ""))
        context_embeds = [embeddings.embed_query(doc.page_content) for doc in response.get("context", []) if hasattr(doc, 'page_content')]
        similarities = [util.cos_sim(answer_embed, ctx_embed).item() for ctx_embed in context_embeds if ctx_embed is not None]
        scores["faithfulness"] = float(np.mean(similarities)) if similarities else 0.0
        scores["hit_rate"] = calculate_hit_rate(
            retriever=None,
            query=response.get("input", ""),
            expected_docs=[doc.page_content for doc in response.get("context", []) if hasattr(doc, 'page_content')],
            k=1
        )
    except Exception as e:
        logger.error(f"Error in evaluate_rag_response: {str(e)}")
        scores["faithfulness"] = 0.0
        scores["hit_rate"] = 0.0
    return scores

def evaluate_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="distilbert-base-uncased",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.load_local("../models/faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Dummy evaluation for DVC tracking
    test_response = {
        "input": "What is the diagnostic path?",
        "answer": "The diagnostic path involves initial assessment.",
        "context": [vector_store.similarity_search("What is the diagnostic path?")[0]]
    }
    
    scores = evaluate_rag_response(test_response, embeddings)
    
    # Save metrics
    os.makedirs("../reports", exist_ok=True)
    with open("../reports/metrics.json", "w") as f:
        json.dump(scores, f, indent=4)
    
    # Log to DVCLive
    with Live() as live:
        live.log_metric("faithfulness", scores["faithfulness"])
        live.log_metric("hit_rate", scores["hit_rate"])
    
    logger.info(f"Evaluation completed: {scores}")

if __name__ == "__main__":
    evaluate_model()