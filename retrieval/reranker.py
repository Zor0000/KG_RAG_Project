# retrieval/reranker.py

from sentence_transformers import CrossEncoder
from typing import List, Dict


# ============================================================
# 🔹 Load MiniLM Cross Encoder
# ============================================================

print("🧠 Loading MiniLM reranker model...")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ============================================================
# 🔹 Rerank Function
# ============================================================

def rerank(query: str, vector_results: List[Dict], top_k: int = 5):

    if not vector_results:
        return []

    # Prepare (query, chunk_text) pairs
    pairs = [
        (query, r["text"])
        for r in vector_results
    ]

    # Predict relevance scores
    scores = reranker_model.predict(pairs)

    # Attach scores
    for i, r in enumerate(vector_results):
        r["rerank_score"] = float(scores[i])

    # Sort by rerank score (descending)
    sorted_results = sorted(
        vector_results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return sorted_results[:top_k]