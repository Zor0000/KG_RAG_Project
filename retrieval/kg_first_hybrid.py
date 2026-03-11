# retrieval/kg_first_hybrid.py

from typing import List, Dict
from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from ingestion.db import get_connection
import json
import os

# ============================================================
# CONFIG
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Neer@j080105"

MODEL_NAME = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_TOP_K = 10

# ============================================================
# INIT
# ============================================================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
minilm = SentenceTransformer(MINILM_MODEL)

# ============================================================
# STEP 1: Extract Structured Signals
# ============================================================

def extract_query_structure(query: str) -> Dict:

    prompt = f"""
Extract structured retrieval signals from this query.

Return JSON with:
- intent (one of: what-is, how-to, why, reference, troubleshooting, comparison, or null)
- topic_keywords (array of important keywords)
- persona (NoCode, LowCode, ProDeveloper, Admin, Architect, or null)

Query:
{query}

Return ONLY valid JSON.
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
        structured = json.loads(raw)
    except:
        structured = {
            "intent": None,
            "topic_keywords": [],
            "persona": None
        }

    print("\n🧠 Extracted Structure:")
    print(structured)

    return structured

# ============================================================
# STEP 2: Build Cypher
# ============================================================

def build_cypher(structured: Dict):

    intent = structured.get("intent")
    keywords = structured.get("topic_keywords", [])
    persona = structured.get("persona")

    where_clauses = []
    params = {}

    query = "MATCH (c:Chunk) "

    if intent:
        query += "MATCH (c)-[:HAS_INTENT]->(i:Intent) "
        where_clauses.append("i.name = $intent")
        params["intent"] = intent

    if persona:
        query += "MATCH (c)-[:HAS_PERSONA]->(p:Persona) "
        where_clauses.append("p.name = $persona")
        params["persona"] = persona

    if keywords:
        query += "MATCH (c)-[:HAS_TOPIC]->(t:Topic) "
        keyword_clauses = []
        for idx, kw in enumerate(keywords):
            key = f"kw{idx}"
            keyword_clauses.append(f"toLower(t.name) CONTAINS toLower(${key})")
            params[key] = kw
        where_clauses.append("(" + " OR ".join(keyword_clauses) + ")")

    if where_clauses:
        query += "WHERE " + " AND ".join(where_clauses) + " "

    query += "RETURN c.chunk_id AS chunk_id LIMIT 50"

    print("\n🔵 Generated Cypher:")
    print(query)
    print("\n🟡 Params:")
    print(params)

    return query, params

# ============================================================
# STEP 3: Run Cypher
# ============================================================

def run_cypher(query: str, params: Dict) -> List[str]:

    with driver.session() as session:
        results = session.run(query, **params)
        chunk_ids = [r["chunk_id"] for r in results]

    print("\n🔎 Sample Neo4j chunk_ids:")
    print(chunk_ids[:5])

    return chunk_ids

# ============================================================
# STEP 4: Fetch from Postgres (hash version)
# ============================================================

def fetch_chunks_from_postgres(chunk_ids: List[str]) -> List[Dict]:

    if not chunk_ids:
        return []

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, chunk_text
        FROM ingestion.chunks
        WHERE id::text = ANY(%s)
    """, (chunk_ids,))

    rows = cur.fetchall()
    conn.close()

    print(f"🟣 Postgres returned {len(rows)} rows")

    return [
        {"chunk_id": str(r[0]), "text": r[1]}
        for r in rows
    ]
# ============================================================
# STEP 5: MiniLM Rerank
# ============================================================

def rerank_with_minilm(query: str, chunks: List[Dict]) -> List[Dict]:

    if not chunks:
        return []

    query_emb = minilm.encode(query, convert_to_tensor=True)
    texts = [c["text"] for c in chunks]
    emb = minilm.encode(texts, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, emb)[0]

    for idx, score in enumerate(scores):
        chunks[idx]["minilm_score"] = float(score)

    chunks.sort(key=lambda x: x["minilm_score"], reverse=True)

    return chunks[:MINILM_TOP_K]

# ============================================================
# MAIN
# ============================================================

def kg_first_search(query: str) -> List[Dict]:

    structured = extract_query_structure(query)
    cypher_query, params = build_cypher(structured)
    chunk_hashes = run_cypher(cypher_query, params)
    chunks = fetch_chunks_from_postgres(chunk_hashes)
    reranked = rerank_with_minilm(query, chunks)

    return reranked

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    while True:
        q = input("\n💬 Ask (or exit): ")

        if q.lower() == "exit":
            break

        results = kg_first_search(q)

        if not results:
            print("\n❌ No results found.\n")
            continue

        print(f"\n🔎 Top {len(results)} Results:\n")

        for i, r in enumerate(results, 1):
            print(f"{i}. Score: {r['minilm_score']:.4f}")
            print("   Text Preview:", r["text"][:200])
            print()