# ingestion/ingest_kg.py

from neo4j import GraphDatabase
from ingestion.db import get_connection


# ============================================================
# CONFIG
# ============================================================

ENRICHMENT_VERSION = 1

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Neer@j080105"
NEO4J_DATABASE = "copilot-kg-v5"  # 🔥 SAME DATABASE


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


# ============================================================
# CONSTRAINTS
# ============================================================

CONSTRAINTS = [
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT super_topic_name IF NOT EXISTS FOR (s:SuperTopic) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT persona_name IF NOT EXISTS FOR (p:Persona) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT intent_name IF NOT EXISTS FOR (i:Intent) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT document_url IF NOT EXISTS FOR (d:Document) REQUIRE d.url IS UNIQUE",
    "CREATE CONSTRAINT product_name IF NOT EXISTS FOR (pr:Product) REQUIRE pr.name IS UNIQUE"
]


# ============================================================
# FETCH ENRICHED CHUNKS
# ============================================================

def fetch_kg_rows(source_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            c.id,
            c.chunk_type,
            ec.confidence,
            ec.complexity,
            ec.canonical_topic,
            ec.super_topic,
            ec.persona,
            ec.intent,
            d.url,
            d.title,
            d.product
        FROM ingestion.chunks c
        JOIN ingestion.enriched_chunks ec
            ON ec.chunk_id = c.id
            AND ec.enrichment_version = %s
        JOIN ingestion.documents d
            ON d.id = c.document_id
        WHERE d.source_id = %s
    """, (ENRICHMENT_VERSION, source_id))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return rows


# ============================================================
# CYPHER
# ============================================================

UPSERT_GRAPH = """
MERGE (c:Chunk {chunk_id: $chunk_id})
SET c.chunk_type = $chunk_type,
    c.confidence = $confidence,
    c.complexity = $complexity

MERGE (t:Topic {name: $canonical_topic})
MERGE (c)-[:HAS_TOPIC]->(t)

MERGE (s:SuperTopic {name: $super_topic})
MERGE (t)-[:BELONGS_TO]->(s)

MERGE (p:Persona {name: $persona})
MERGE (c)-[:HAS_PERSONA]->(p)

MERGE (i:Intent {name: $intent})
MERGE (c)-[:HAS_INTENT]->(i)

MERGE (d:Document {url: $url})
SET d.title = $title
MERGE (c)-[:FROM_DOCUMENT]->(d)

MERGE (pr:Product {name: $product})
MERGE (c)-[:BELONGS_TO_PRODUCT]->(pr)
"""


# ============================================================
# MAIN
# ============================================================

def main(source_id):

    rows = fetch_kg_rows(source_id)
    print(f"\n🧠 Ingesting {len(rows)} chunks into Neo4j (source: {source_id})...\n")

    if not rows:
        print("⚠️ No enriched chunks found.")
        return

    with driver.session(database=NEO4J_DATABASE) as session:

        # Ensure constraints exist
        for constraint in CONSTRAINTS:
            session.run(constraint)

        for row in rows:

            (
                chunk_id,
                chunk_type,
                confidence,
                complexity,
                canonical_topic,
                super_topic,
                persona,
                intent,
                url,
                title,
                product
            ) = row

            session.run(
                UPSERT_GRAPH,
                chunk_id=str(chunk_id),
                chunk_type=chunk_type or "unknown",
                confidence=float(confidence or 0.0),
                complexity=complexity or "unknown",
                canonical_topic=canonical_topic or "Unknown",
                super_topic=super_topic or "General",
                persona=persona or "Unknown",
                intent=intent or "reference",
                url=url,
                title=title or "",
                product=product or "unknown"
            )

    print("🎉 Neo4j KG ingestion complete\n")