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

VECTOR_TOP_K = 120
RERANK_TOP_K = 10
EXPANSION_LIMIT = 50


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
# STEP 2 — Graph Topic Expansion
# ============================================================

def expand_topics(seed_topics):

    if not seed_topics:
        return []

    with driver.session(database=NEO4J_DB) as session:

        res = session.run("""
        MATCH (t:Topic)
        WHERE t.name IN $topics
        OPTIONAL MATCH (t)-[:RELATED_TO]-(related:Topic)
        OPTIONAL MATCH (d:Document)-[:COVERS_TOPIC]->(t)
        RETURN DISTINCT coalesce(related.name, t.name) AS topic
        LIMIT $limit
        """,
        {"topics": seed_topics, "limit": EXPANSION_LIMIT}
        )

        expanded = [r["topic"] for r in res]

    expanded = list(set(seed_topics + expanded))

    print("\n🌐 Expanded Topics:", expanded)

    return expanded


# ============================================================
# STEP 3 — OpenAI Embedding
# ============================================================

def embed_query(query):

    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )

    return response.data[0].embedding


# ============================================================
# STEP 4 — Milvus Search
# ============================================================

def search_milvus(query):

    query_vector = embed_query(query)

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={
            "metric_type": "COSINE",
            "params": {"ef": 200}
        },
        limit=VECTOR_TOP_K,
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
            "vector_score": hit.distance
        })

    return hits


# ============================================================
# STEP 5 — Extract Topics From Chunks
# ============================================================

def extract_topics_from_chunks(chunks):

    topics = set()

    for c in chunks:
        if c.get("topic"):
            topics.add(c["topic"])

    topics = list(topics)

    print("\n🔎 Topics from chunks:", topics[:10])

    return topics


# ============================================================
# STEP 6 — Graph Boost
# ============================================================

def graph_boost(chunks, expanded_topics):

    expanded_set = set(expanded_topics)

    for c in chunks:

        if c["topic"] in expanded_set:
            c["graph_bonus"] = 0.1
        else:
            c["graph_bonus"] = 0

    return chunks


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

        graph_bonus = chunks[i].get("graph_bonus", 0)

        chunks[i]["rerank_score"] = float(score) + graph_bonus

    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    return chunks[:RERANK_TOP_K]


# ============================================================
# MAIN PIPELINE
# ============================================================

def graph_guided_search(query, persona="All", product="All", top_k=5):

    print("\n==============================")
    print("🔎 QUERY:", query)
    print("==============================")

    structure = extract_query_structure(query)

    # Step 1 — Vector retrieval
    milvus_hits = search_milvus(query)

    # Step 2 — Extract topics from retrieved chunks
    chunk_topics = extract_topics_from_chunks(milvus_hits)

    # Step 3 — Graph expansion
    expanded_topics = expand_topics(chunk_topics)

    # Step 4 — Graph boost
    milvus_hits = graph_boost(milvus_hits, expanded_topics)

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

    # Step 5 — Rerank
    reranked = rerank(query, milvus_hits)

    # Step 6 — Deduplicate topics
    seen = set()
    unique = []

    for r in reranked:

        if r["topic"] not in seen:
            unique.append(r)
            seen.add(r["topic"])

    return {
        "results": unique[:top_k],
        "structure": structure,
        "detected_topics": chunk_topics,
        "expanded_topics": expanded_topics
    }

# ============================================================
# LANGSERVE ENTRYPOINT
# ============================================================

def retrieve_answer(question: str):
    """
    LangServe-compatible wrapper.
    This is the function the API will call.
    """

    results = graph_guided_search(question)

    if not results["results"]:
        return {
            "answer": "No relevant information found.",
            "sources": []
        }

    context = "\n\n".join(
        [r["text"] for r in results["results"]]
    )

    prompt = f"""
Use the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and cite relevant information if needed.
"""

    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful Copilot Studio expert."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": results["results"],
        "structure": results["structure"],
        "topics": results["expanded_topics"]
    }
# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    while True:

        q = input("\n💬 Ask (or type exit): ")

        if q.lower() == "exit":
            break

        results = graph_guided_search(q)

        if not results["results"]:
            print("\n❌ No results\n")
            continue

        print("\n🏆 Top Results\n")

        for i, r in enumerate(results["results"], 1):

            print(f"{i}. Score: {r['rerank_score']:.4f}")
            print(f"Topic: {r['topic']}")
            print(r["text"][:300])
            print()