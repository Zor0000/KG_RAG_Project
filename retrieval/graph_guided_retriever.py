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

LLM_MODEL = "gpt-5.2"

VECTOR_TOP_K = 80
RERANK_TOP_K = 8
TOPIC_LIMIT = 40

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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

intent: one of
- how-to
- troubleshooting
- reference
- comparison
- explanation

persona: one of
- NoCode
- LowCode
- ProDeveloper
- Admin
- Architect

product: if mentioned

keywords: array

Query:
{query}

Return ONLY JSON.
"""

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "You extract structured metadata."},
            {"role": "user", "content": prompt}
        ]
    )

    raw = response.choices[0].message.content.strip()

    try:
        return json.loads(raw)
    except:
        return {
            "intent": None,
            "persona": None,
            "product": None,
            "keywords": []
        }


# ============================================================
# STEP 2 — KG Topic Routing
# ============================================================

def kg_topic_routing(intent, persona, product):

    with driver.session(database=NEO4J_DB) as session:

        query = """
        MATCH (t:Topic)

        OPTIONAL MATCH (t)<-[:SUPPORTS_INTENT]-(i:Intent)
        OPTIONAL MATCH (t)<-[:TARGETS_PERSONA]-(p:Persona)
        OPTIONAL MATCH (t)<-[:USED_IN_PRODUCT]-(prod:Product)

        WHERE
            ($intent IS NULL OR i.name = $intent)
        AND ($persona IS NULL OR p.name = $persona)
        AND ($product IS NULL OR prod.name = $product)

        RETURN DISTINCT t.name AS topic
        LIMIT $limit
        """

        res = session.run(query, {
            "intent": intent,
            "persona": persona,
            "product": product,
            "limit": TOPIC_LIMIT
        })

        topics = [r["topic"] for r in res]

    print("\n🎯 Routed Topics:", topics[:10])

    return topics


# ============================================================
# STEP 3 — Topic Expansion
# ============================================================

def expand_topics(seed_topics):

    if not seed_topics:
        return []

    with driver.session(database=NEO4J_DB) as session:

        res = session.run("""
        MATCH (t:Topic)
        WHERE t.name IN $topics

        OPTIONAL MATCH (t)-[:RELATED_TO]-(related:Topic)
        OPTIONAL MATCH (super:SuperTopic)-[:HAS_TOPIC]->(t)

        RETURN DISTINCT
        coalesce(related.name, t.name) AS topic
        LIMIT $limit
        """,
        {
            "topics": seed_topics,
            "limit": TOPIC_LIMIT
        })

        expanded = [r["topic"] for r in res]

    expanded = list(set(seed_topics + expanded))

    print("\n🌐 Expanded Topics:", expanded[:10])

    return expanded


# ============================================================
# STEP 4 — Query Embedding
# ============================================================

def embed_query(query):

    res = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )

    return res.data[0].embedding


# ============================================================
# STEP 5 — Topic Constrained Vector Search
# ============================================================

def vector_search(query, allowed_topics):

    vector = embed_query(query)

    expr = None

    if allowed_topics:
        topic_list = ",".join([f'"{t}"' for t in allowed_topics])
        expr = f'canonical_topic in [{topic_list}]'

    results = collection.search(
        data=[vector],
        anns_field="embedding",
        param={
            "metric_type": "COSINE",
            "params": {"ef": 200}
        },
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

        e = hit.entity

        hits.append({
            "chunk_id": e.get("chunk_id"),
            "text": e.get("text"),
            "topic": e.get("canonical_topic"),
            "persona": e.get("persona"),
            "intent": e.get("intent"),
            "product": e.get("product"),
            "vector_score": hit.distance
        })

    return hits


# ============================================================
# STEP 6 — Rerank
# ============================================================

def rerank(query, chunks):

    if not chunks:
        return []

    texts = [c["text"] for c in chunks]

    q_emb = minilm.encode(query, convert_to_tensor=True)
    d_emb = minilm.encode(texts, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, d_emb)[0]

    for i, s in enumerate(scores):

        chunks[i]["rerank_score"] = float(s)

    chunks.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return chunks[:RERANK_TOP_K]


# ============================================================
# MAIN PIPELINE
# ============================================================

def graph_guided_search(query, persona="All", product="All", top_k=5):

    print("\n==============================")
    print("🔎 QUERY:", query)
    print("==============================")

    structure = extract_query_structure(query)

    intent = structure.get("intent")

    if persona == "All":
        persona = structure.get("persona")

    if product == "All":
        product = structure.get("product")

    seed_topics = kg_topic_routing(intent, persona, product)

    expanded_topics = expand_topics(seed_topics)

    chunks = vector_search(query, expanded_topics)

    results = rerank(query, chunks)

    return {
        "results": results[:top_k],
        "topics": expanded_topics,
        "structure": structure
    }


# ============================================================
# ANSWER GENERATION
# ============================================================

def retrieve_answer(question):

    retrieval = graph_guided_search(question)

    if not retrieval["results"]:
        return {"answer": "No relevant information found."}

    context = "\n\n".join(
        [r["text"] for r in retrieval["results"]]
    )

    prompt = f"""
Use the context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly.
"""

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a Copilot Studio expert."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": retrieval["results"],
        "topics": retrieval["topics"],
        "structure": retrieval["structure"]
    }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    while True:

        q = input("\n💬 Ask (exit to quit): ")

        if q.lower() == "exit":
            break

        res = retrieve_answer(q)

        print("\n🧠 Answer:\n")
        print(res["answer"])

        print("\n📚 Sources:\n")

        for r in res["sources"]:
            print("-", r["topic"])