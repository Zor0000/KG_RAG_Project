# retrieval/graph_guided_retriever.py

import os
import json
from typing import List, Dict

from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from pymilvus import connections, Collection


# ============================================================
# CONFIG
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Neer@j080105"
NEO4J_DB = "copilot-kg-v6"

MILVUS_COLLECTION = "project_chunks_v5"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-5.2"

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOPIC_DETECTION_K = 5
EXPANSION_LIMIT = 50
VECTOR_TOP_K = 40
RERANK_TOP_K = 10


# ============================================================
# INIT CLIENTS
# ============================================================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

minilm = SentenceTransformer(MINILM_MODEL)

connections.connect("default", host="localhost", port="19530")
collection = Collection(MILVUS_COLLECTION)


# ============================================================
# STEP 1 — Extract Query Structure
# ============================================================

def extract_query_structure(query: str) -> Dict:

    prompt = f"""
Extract structured signals from the query.

Return JSON with:
- intent (what-is, how-to, why, reference, troubleshooting, comparison or null)
- persona (NoCode, LowCode, ProDeveloper, Admin, Architect or null)
- keywords (array)

Query:
{query}

Return ONLY JSON.
"""

    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": "You extract structured metadata."},
            {"role": "user", "content": prompt}
        ]
    )

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except:
        data = {"intent": None, "persona": None, "keywords": []}

    print("\n🧠 Query Structure:", data)

    return data


# ============================================================
# STEP 2 — Fetch All Topics
# ============================================================

def fetch_all_topics():

    with driver.session(database=NEO4J_DB) as session:

        res = session.run("""
        MATCH (t:Topic)
        RETURN t.name AS name
        """)

        topics = [r["name"] for r in res]

    return topics


# ============================================================
# STEP 3 — Detect Topics
# ============================================================

def detect_topics(query: str):

    topics = fetch_all_topics()

    topic_emb = minilm.encode(topics, convert_to_tensor=True)
    query_emb = minilm.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, topic_emb)[0]

    top_idx = scores.topk(TOPIC_DETECTION_K).indices

    detected = [topics[i] for i in top_idx]

    print("\n🔎 Detected Topics:", detected)

    return detected


# ============================================================
# STEP 4 — Expand Topics
# ============================================================

def expand_topics(seed_topics):

    with driver.session(database=NEO4J_DB) as session:

        res = session.run("""
        MATCH (t:Topic)
        WHERE t.name IN $topics
        MATCH (t)-[:HAS_SUPER_TOPIC]->(s:SuperTopic)
        MATCH (s)<-[:HAS_SUPER_TOPIC]-(related:Topic)
        RETURN DISTINCT related.name AS topic
        LIMIT $limit
        """,
        {"topics": seed_topics, "limit": EXPANSION_LIMIT}
        )

        expanded = [r["topic"] for r in res]

    if not expanded:
        expanded = seed_topics

    expanded = list(set(seed_topics + expanded))

    print("\n🌐 Expanded Topics:", expanded)

    return expanded


# ============================================================
# STEP 5 — OpenAI Embedding
# ============================================================

def embed_query(query):

    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )

    return response.data[0].embedding


# ============================================================
# STEP 6 — Milvus Search
# ============================================================

def search_milvus(query, topics):

    if not topics:
        return []

    query_vector = embed_query(query)

    expr = f'canonical_topic in {topics}'

    print("\n📦 Milvus Filter:", expr)

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 100}},
        limit=VECTOR_TOP_K,
        expr=expr,
        output_fields=[
            "chunk_id",
            "text",
            "canonical_topic",
            "persona",
            "intent",
            "product"
        ]
    )

    hits = []

    for hit in results[0]:

        entity = hit.entity

        hits.append({
            "chunk_id": entity.get("chunk_id"),
            "text": entity.get("text"),
            "topic": entity.get("canonical_topic"),
            "persona": entity.get("persona"),
            "intent": entity.get("intent"),
            "product": entity.get("product"),
            "score": hit.distance
        })

    return hits


# ============================================================
# STEP 7 — Reranking
# ============================================================

def rerank(query, chunks):

    if not chunks:
        return []

    texts = [c["text"] for c in chunks]

    query_emb = minilm.encode(query, convert_to_tensor=True)
    doc_emb = minilm.encode(texts, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, doc_emb)[0]

    for i, score in enumerate(scores):
        chunks[i]["rerank_score"] = float(score)

    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    return chunks[:RERANK_TOP_K]


# ============================================================
# MAIN PIPELINE
# ============================================================

def graph_guided_search(query, persona="All", product="All", top_k=5):

    structure = extract_query_structure(query)

    detected_topics = detect_topics(query)

    expanded_topics = expand_topics(detected_topics)

    milvus_hits = search_milvus(query, expanded_topics)

    # -----------------------------
    # Persona Filter
    # -----------------------------

    if persona != "All":
        milvus_hits = [
            c for c in milvus_hits
            if c.get("persona") == persona
        ]

    # -----------------------------
    # Product Filter
    # -----------------------------

    if product != "All":
        milvus_hits = [
            c for c in milvus_hits
            if c.get("product") == product
        ]

    reranked = rerank(query, milvus_hits)

    # -----------------------------
    # Deduplicate topics
    # -----------------------------

    seen = set()
    unique = []

    for r in reranked:
        if r["topic"] not in seen:
            unique.append(r)
            seen.add(r["topic"])

    return {
        "results": unique[:top_k],
        "structure": structure,
        "detected_topics": detected_topics,
        "expanded_topics": expanded_topics
    }