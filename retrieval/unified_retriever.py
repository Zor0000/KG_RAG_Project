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
# 🔹 Product keyword map  (expanded)
# ============================================================
# General Microsoft / Copilot / business queries always map to
# copilot_studio so they never trigger false clarifications.

PRODUCT_KEYWORDS: Dict[str, List[str]] = {
    "copilot_studio": [
        "copilot studio", "copilot", "microsoft copilot",
        "licensing", "license", "m365", "microsoft 365",
        "teams", "outlook", "sharepoint", "power platform",
        "power automate", "power apps", "power bi",
        "business developer", "business user", "business development",
        "lab", "labs", "hands-on", "exercise", "tutorial",
        "learning path", "training", "workshop", "scenario",
        "relevant", "recommend", "best for", "get started",
        "beginner", "intermediate", "advanced",
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


# ============================================================
# 🔹 Lab query detection
# ============================================================
# Detects when the user is asking specifically about labs,
# exercises, learning paths, or recommendations — so we can
# apply a lab-specific retrieval boost and answer strategy.

LAB_QUERY_KEYWORDS = [
    "lab", "labs", "exercise", "exercises", "hands-on",
    "tutorial", "tutorials", "learning path", "training",
    "workshop", "scenario", "practice", "project",
    "relevant", "recommend", "best lab", "which lab",
    "get started", "where do i start", "beginner lab",
    "what should i learn", "what labs", "most relevant",
]


def is_lab_query(query: str) -> bool:
    """Returns True if the query is asking about labs or learning resources."""
    q = query.lower()
    return any(kw in q for kw in LAB_QUERY_KEYWORDS)


# ============================================================
# 🔹 Product inference
# ============================================================

def infer_product_from_query(query: str) -> Optional[str]:
    """
    Scan the query for product-specific keywords.
    Returns the inferred product name or None if ambiguous.
    """
    q = query.lower()
    scores: Dict[str, int] = {p: 0 for p in PRODUCT_KEYWORDS}

    for product, keywords in PRODUCT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[product] += 1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

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

def detect_ambiguity(results: List[Dict]):

    products = [r["product"] for r in results if r.get("product")]

    if not products:
        return False, None

    counts  = Counter(products)
    total   = sum(counts.values())

    dominant_product, freq = counts.most_common(1)[0]

    if freq / total > 0.75:
        return False, dominant_product

    if len(counts) >= 2:
        return True, list(counts.keys())

    return False, dominant_product


# ============================================================
# 🔹 Fallback Search
# ============================================================
# Called when the primary search returns empty or all results
# score below the quality threshold.
# Strategy:
#   1. Broaden the search — remove product filter, raise top_k
#   2. Re-run vector search with a simplified query
#   3. If still empty → return structured "no results" mode
#      so answer_generator can give a helpful response
#      instead of a dead "I don't have info" answer.

FALLBACK_SCORE_THRESHOLD = 0.30   # below this = weak result


def _simplify_query(query: str) -> str:
    """
    Strip filler words to create a shorter fallback query.
    E.g. 'Which labs are most relevant for me as a business developer'
         → 'labs business developer Microsoft Copilot'
    """
    stopwords = {
        "which", "what", "are", "the", "most", "for", "me", "as", "a",
        "an", "is", "i", "my", "can", "you", "do", "have", "using",
        "with", "in", "on", "of", "to", "and", "or", "at", "be",
        "this", "that", "these", "those", "how", "should", "would",
        "could", "please", "tell", "give", "show", "find",
    }
    words   = query.lower().split()
    cleaned = [w for w in words if w not in stopwords]
    return " ".join(cleaned) if cleaned else query


def fallback_search(query: str, persona: Optional[str], top_k: int) -> dict:
    """
    Broadened search with no product filter and a simplified query.
    Returns unified_search-style response dict.
    """
    simplified = _simplify_query(query)
    print(f"🔄 Fallback search: '{simplified}'")

    results = vector_search(
        query          = simplified,
        top_k          = top_k * 4,    # cast a wider net
        persona_filter = persona if persona and persona != "All" else None,
        product_filter = None,          # no product filter in fallback
    )

    if not results:
        return {"mode": "no_results", "query": query}

    # Accept any result above a lower threshold
    good = [r for r in results if r.get("score", 0) >= FALLBACK_SCORE_THRESHOLD]

    if not good:
        return {"mode": "no_results", "query": query}

    # Rerank and return
    seed_ids  = [r["chunk_id"] for r in good[:5]]
    kg_scores = kg_expand(seed_ids)
    ranked    = rerank_results(good, kg_scores)
    final     = rerank(query, ranked, top_k=top_k)

    return {
        "mode":       "answer",
        "results":    final,
        "is_fallback": True,   # flag so answer_generator can note lower confidence
    }


# ============================================================
# 🔹 Unified Search
# ============================================================

# Results with ALL scores below this threshold are treated as
# weak — triggers fallback rather than generating a bad answer.
QUALITY_THRESHOLD = 0.35


def unified_search(
    query:   str,
    persona: Optional[str] = None,
    product: Optional[str] = None,
    top_k:   int = 5
):
    print("🔎 Running global vector search...")

    # ── Step 1: Detect if this is a lab/recommendation query ─────────
    lab_query = is_lab_query(query)
    if lab_query:
        print("📚 Lab query detected — boosting lab content retrieval")

    # ── Step 2: Hard product filter from UI selectbox ─────────────────
    manual_product  = product if product and product != "All" else None

    # ── Step 3: Keyword-based product inference ───────────────────────
    inferred_product = None
    if not manual_product:
        inferred_product = infer_product_from_query(query)
        if inferred_product:
            print(f"🧠 Product inferred from query: {inferred_product}")

    effective_product = manual_product or inferred_product

    # ── Step 4: Primary vector search ────────────────────────────────
    # For lab queries, raise top_k to surface more lab chunks
    search_top_k = top_k * 5 if lab_query else top_k * 3

    vector_results = vector_search(
        query          = query,
        top_k          = search_top_k,
        persona_filter = persona if persona and persona != "All" else None,
        product_filter = effective_product,
    )

    # ── Step 5: Empty result → fallback immediately ───────────────────
    if not vector_results:
        print("⚠️ No results — running fallback search")
        return fallback_search(query, persona, top_k)

    # ── Step 6: Quality check — if all results are weak → fallback ────
    top_score = max(r.get("score", 0) for r in vector_results)
    if top_score < QUALITY_THRESHOLD:
        print(f"⚠️ Weak results (top score: {top_score:.3f}) — running fallback")
        fallback = fallback_search(query, persona, top_k)
        # If fallback found something better, use it
        if fallback.get("mode") == "answer":
            fallback_top = max(r.get("score", 0) for r in fallback["results"])
            if fallback_top > top_score:
                return fallback
        # Otherwise continue with original weak results
        # (still better than nothing — answer_generator will handle tone)

    # ── Step 7: If product is known → skip ambiguity check ───────────
    if effective_product:
        seed_ids        = [r["chunk_id"] for r in vector_results[:5]]
        kg_scores       = kg_expand(seed_ids)
        metadata_ranked = rerank_results(vector_results, kg_scores)
        final_results   = rerank(query, metadata_ranked, top_k=top_k)

        return {
            "mode":    "answer",
            "results": final_results,
        }

    # ── Step 8: No product known → ambiguity detection ───────────────
    ambiguous, info = detect_ambiguity(vector_results)

    if ambiguous:
        print("⚠️ Ambiguous query — returning preview + options")
        preview_chunks = rerank(query, vector_results, top_k=5)
        return {
            "mode":           "guided_clarification",
            "preview_chunks": preview_chunks,
            "options":        info,
        }

    # ── Step 9: Single dominant product ──────────────────────────────
    dominant_product = info
    print(f"🎯 Dominant product: {dominant_product}")

    filtered        = [r for r in vector_results if r["product"] == dominant_product]
    seed_ids        = [r["chunk_id"] for r in filtered[:5]]
    kg_scores       = kg_expand(seed_ids)
    metadata_ranked = rerank_results(filtered, kg_scores)
    final_results   = rerank(query, metadata_ranked, top_k=top_k)

    return {
        "mode":    "answer",
        "results": final_results,
    }