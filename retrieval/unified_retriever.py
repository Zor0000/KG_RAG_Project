# retrieval/unified_retriever.py

from typing import List, Dict, Optional
from retrieval.vector_retriever import vector_search
from retrieval.reranker import rerank
from neo4j import GraphDatabase
from collections import defaultdict, Counter


# ============================================================
# 🔹 Neo4j Config
# ============================================================

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "Neer@j080105"

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


# ============================================================
# 🔹 Product keyword map
# ============================================================
# If ANY of these keywords appear in the query we can infer
# the product WITHOUT asking the user — avoids false clarifications
# on general Copilot / Microsoft questions.

PRODUCT_KEYWORDS: Dict[str, List[str]] = {
    "copilot_studio": [
        "copilot studio", "copilot", "microsoft copilot",
        "licensing", "license", "m365", "microsoft 365",
        "teams", "outlook", "sharepoint", "power platform",
        "power automate", "power apps", "power bi",
    ],
    "azure_bot_service": [
        "azure bot", "bot service", "bot framework",
        "azure", "direct line", "bot channel",
    ],
    "autogen": [
        "autogen", "multi-agent", "agent framework",
        "agentic", "agent orchestration",
    ],
}


def infer_product_from_query(query: str) -> Optional[str]:
    """
    Scan the query for product-specific keywords.
    Returns the inferred product name or None if ambiguous.

    Scoring: count keyword hits per product, return the winner
    only if it scores strictly higher than all others.
    """
    q = query.lower()
    scores: Dict[str, int] = {p: 0 for p in PRODUCT_KEYWORDS}

    for product, keywords in PRODUCT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[product] += 1

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Only infer if top product has a clear lead and at least 1 hit
    if ranked[0][1] > 0 and (len(ranked) < 2 or ranked[0][1] > ranked[1][1]):
        return ranked[0][0]

    return None


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
        cid          = h["chunk_id"]
        vector_score = h["score"]
        kg           = kg_scores.get(cid, {})

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
# Threshold raised from 0.65 → 0.75 to reduce false clarifications.
# A product needs 75% of top results to be considered dominant.

def detect_ambiguity(results: List[Dict]):

    products = [r["product"] for r in results if r.get("product")]

    if not products:
        return False, None

    counts  = Counter(products)
    total   = sum(counts.values())

    dominant_product, freq = counts.most_common(1)[0]

    # Strong dominance — no clarification needed
    if freq / total > 0.75:
        return False, dominant_product

    # Multiple products with no clear winner
    if len(counts) >= 2:
        return True, list(counts.keys())

    return False, dominant_product


# ============================================================
# 🔹 Unified Search
# ============================================================

def unified_search(
    query:   str,
    persona: Optional[str] = None,
    product: Optional[str] = None,
    top_k:   int = 5
):
    print("🔎 Running global vector search...")

    # ── Step 1: Hard product filter from UI selectbox ────────────────
    manual_product = product if product and product != "All" else None

    # ── Step 2: Keyword-based product inference from query ───────────
    # Runs even when product="All" so general Copilot/M365 questions
    # don't falsely trigger guided clarification.
    inferred_product = None
    if not manual_product:
        inferred_product = infer_product_from_query(query)
        if inferred_product:
            print(f"🧠 Product inferred from query: {inferred_product}")

    # Effective product to filter on (manual takes priority over inferred)
    effective_product = manual_product or inferred_product

    vector_results = vector_search(
        query          = query,
        top_k          = 30,
        persona_filter = persona if persona and persona != "All" else None,
        product_filter = effective_product,
    )

    if not vector_results:
        return {"mode": "empty"}

    # ── Step 3: If product is known (manual or inferred) → skip ambiguity
    if effective_product:
        seed_ids      = [r["chunk_id"] for r in vector_results[:5]]
        kg_scores     = kg_expand(seed_ids)
        metadata_ranked = rerank_results(vector_results, kg_scores)
        final_results = rerank(query, metadata_ranked, top_k=top_k)

        return {
            "mode":    "answer",
            "results": final_results,
        }

    # ── Step 4: No product known → run ambiguity detection ───────────
    ambiguous, info = detect_ambiguity(vector_results)

    if ambiguous:
        print("⚠️ Ambiguous query detected. Returning preview + options.")
        preview_chunks = rerank(query, vector_results, top_k=5)
        return {
            "mode":           "guided_clarification",
            "preview_chunks": preview_chunks,
            "options":        info,
        }

    # ── Step 5: Single dominant product path ─────────────────────────
    dominant_product = info
    print(f"🎯 Dominant product detected: {dominant_product}")

    filtered      = [r for r in vector_results if r["product"] == dominant_product]
    seed_ids      = [r["chunk_id"] for r in filtered[:5]]
    kg_scores     = kg_expand(seed_ids)
    metadata_ranked = rerank_results(filtered, kg_scores)
    final_results = rerank(query, metadata_ranked, top_k=top_k)

    return {
        "mode":    "answer",
        "results": final_results,
    }