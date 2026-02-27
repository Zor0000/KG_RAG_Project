# retrieval/unified_retriever.py

from typing import List, Dict, Optional
from retrieval.vector_retriever import vector_search
from retrieval.reranker import rerank
from neo4j import GraphDatabase
from collections import defaultdict, Counter


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


# ============================================================
# 🔹 KG Expansion
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

    try:
        with driver.session() as session:

            for r in session.run(super_query, seed_ids=seed_chunk_ids):
                expansion_scores[r["cid"]]["super_overlap"] = r["overlap"]

            for r in session.run(topic_query, seed_ids=seed_chunk_ids):
                expansion_scores[r["cid"]]["topic_overlap"] = r["overlap"]

            for r in session.run(intent_query, seed_ids=seed_chunk_ids):
                expansion_scores[r["cid"]]["intent_overlap"] = r["overlap"]

    except Exception as e:
        print("⚠️ KG unavailable:", e)

    return expansion_scores


# ============================================================
# 🔹 Metadata Fusion
# ============================================================

def rerank_results(base_hits: List[Dict], kg_scores: Dict[str, Dict]) -> List[Dict]:

    for h in base_hits:
        cid = h["chunk_id"]
        vector_score = h["score"]
        kg = kg_scores.get(cid, {})

        final_score = (
            0.60 * vector_score
            + 0.15 * (min(kg.get("super_overlap", 0), 3) / 3)
            + 0.10 * (min(kg.get("topic_overlap", 0), 3) / 3)
            + 0.05 * (min(kg.get("intent_overlap", 0), 3) / 3)
            + 0.10 * (h.get("confidence", 0.5))
        )

        h["final_score"] = final_score

    base_hits.sort(key=lambda x: x["final_score"], reverse=True)
    return base_hits


# ============================================================
# 🔹 Ambiguity Detection
# ============================================================

def detect_ambiguity(results: List[Dict]):

    products = [r["product"] for r in results if r.get("product")]

    if not products:
        return False, None

    counts = Counter(products)
    total = sum(counts.values())

    dominant_product, freq = counts.most_common(1)[0]

    # Strong dominance
    if freq / total > 0.65:
        return False, dominant_product

    # Multiple products present
    if len(counts) >= 2:
        return True, list(counts.keys())

    return False, dominant_product


# ============================================================
# 🔹 Unified Search
# ============================================================

def unified_search(
    query: str,
    persona: Optional[str] = None,
    product: Optional[str] = None,
    top_k: int = 5
):

    print("🔎 Running global vector search...")

    # If user manually selects product → hard filter
    manual_product = product if product and product != "All" else None

    vector_results = vector_search(
        query=query,
        top_k=30,
        persona_filter=persona,
        product_filter=manual_product
    )

    if not vector_results:
        return {"mode": "empty"}

    # If manual product override → skip ambiguity logic
    if manual_product:
        seed_ids = [r["chunk_id"] for r in vector_results[:5]]
        kg_scores = kg_expand(seed_ids)
        metadata_ranked = rerank_results(vector_results, kg_scores)
        final_results = rerank(query, metadata_ranked, top_k=top_k)

        return {
            "mode": "answer",
            "results": final_results
        }

    # ===============================
    # Ambiguity Detection
    # ===============================

    ambiguous, info = detect_ambiguity(vector_results)

    # If ambiguous → guided clarification mode
    if ambiguous:

        print("⚠️ Ambiguous query detected. Returning preview + options.")

        # Use top mixed chunks for preview
        preview_chunks = rerank(query, vector_results, top_k=5)

        return {
            "mode": "guided_clarification",
            "preview_chunks": preview_chunks,
            "options": info
        }

    # ===============================
    # Dominant Product Path
    # ===============================

    dominant_product = info

    print(f"🎯 Dominant product detected: {dominant_product}")

    filtered = [
        r for r in vector_results
        if r["product"] == dominant_product
    ]

    seed_ids = [r["chunk_id"] for r in filtered[:5]]
    kg_scores = kg_expand(seed_ids)

    metadata_ranked = rerank_results(filtered, kg_scores)
    final_results = rerank(query, metadata_ranked, top_k=top_k)

    return {
        "mode": "answer",
        "results": final_results
    }