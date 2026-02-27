# retrieval/hybrid_retriever.py

from retrieval.vector_retriever import vector_search
from neo4j import GraphDatabase
from typing import List, Dict
from collections import defaultdict
import os


# ============================================================
# 🔹 Neo4j Config
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Neer@j080105"

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

def detect_product(query: str) -> str | None:

    q = query.lower()

    if "azure bot" in q or "bot framework" in q:
        return "azure_bot_service"

    if "copilot" in q:
        return "copilot_studio"

    if "autogen" in q:
        return "autogen"

    return None

# ============================================================
# 🔹 KG Expansion (Improved)
# ============================================================

def kg_expand(seed_chunk_ids: List[str]) -> Dict[str, Dict]:

    expansion_scores = defaultdict(lambda: {
        "super_overlap": 0,
        "topic_overlap": 0,
        "intent_overlap": 0,
    })

    super_query = """
    MATCH (c:Chunk)-[:HAS_TOPIC]->(:Topic)-[:BELONGS_TO]->(s:SuperTopic)
          <-[:BELONGS_TO]-(:Topic)<-[:HAS_TOPIC]-(other:Chunk)
    WHERE c.chunk_id IN $seed_ids
    RETURN other.chunk_id AS cid, count(s) AS overlap
    """

    topic_query = """
    MATCH (c:Chunk)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(other:Chunk)
    WHERE c.chunk_id IN $seed_ids
    RETURN other.chunk_id AS cid, count(t) AS overlap
    """

    intent_query = """
    MATCH (c:Chunk)-[:HAS_INTENT]->(i:Intent)<-[:HAS_INTENT]-(other:Chunk)
    WHERE c.chunk_id IN $seed_ids
    RETURN other.chunk_id AS cid, count(i) AS overlap
    """

    with driver.session() as session:

        for r in session.run(super_query, seed_ids=seed_chunk_ids):
            expansion_scores[r["cid"]]["super_overlap"] = r["overlap"]

        for r in session.run(topic_query, seed_ids=seed_chunk_ids):
            expansion_scores[r["cid"]]["topic_overlap"] = r["overlap"]

        for r in session.run(intent_query, seed_ids=seed_chunk_ids):
            expansion_scores[r["cid"]]["intent_overlap"] = r["overlap"]

    return expansion_scores


# ============================================================
# 🔹 Score Fusion
# ============================================================

def rerank_results(
    base_hits: List[Dict],
    kg_scores: Dict[str, Dict],
) -> List[Dict]:

    reranked = []

    for h in base_hits:

        cid = h["chunk_id"]
        vector_score = h["score"]

        kg = kg_scores.get(cid, {})
        super_overlap = min(kg.get("super_overlap", 0), 3) / 3
        topic_overlap = min(kg.get("topic_overlap", 0), 3) / 3
        intent_overlap = min(kg.get("intent_overlap", 0), 3) / 3

        final_score = (
            0.60 * vector_score
            + 0.15 * super_overlap
            + 0.10 * topic_overlap
            + 0.05 * intent_overlap
            + 0.10 * (h.get("confidence", 0.5))
        )

        h["final_score"] = final_score
        reranked.append(h)

    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked


# ============================================================
# 🔹 Hybrid Search
# ============================================================

from retrieval.reranker import rerank

def hybrid_search(query: str, top_k: int = 6):

    product_filter = detect_product(query)

    # Vector
    vector_results = vector_search(
        query,
        top_k=30,
        product_filter=product_filter
    )

    if not vector_results:
        return []

    # KG
    seed_ids = [r["chunk_id"] for r in vector_results[:5]]
    kg_scores = kg_expand(seed_ids)

    metadata_ranked = rerank_results(vector_results, kg_scores)

    # 🔥 MiniLM semantic rerank
    final_results = rerank(query, metadata_ranked, top_k=top_k)

    return final_results


# ============================================================
# 🔹 CLI Test
# ============================================================

if __name__ == "__main__":

    while True:
        q = input("\n💬 Enter query (or 'exit'): ")

        if q.lower() == "exit":
            break

        results = hybrid_search(q)

        print("\n🔥 Hybrid Results:\n")

        for i, r in enumerate(results, 1):
            print(f"{i}. Score: {r['final_score']:.4f}")
            print(f"   Product: {r['product']}")
            print(f"   Persona: {r['persona']}")
            print(f"   Topic: {r['canonical_topic']}")
            print("   Text:", r["text"][:200], "\n")