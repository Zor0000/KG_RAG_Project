# ingestion/ingest_kg.py
import sys

# ingestion/ingest_kg_v6.py
import sys
from neo4j import GraphDatabase
from ingestion.db import get_connection


# ============================================================
# CONFIG
# ============================================================

ENRICHMENT_VERSION = 1

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"  # 🔥 SAME DATABASE
NEO4J_PASSWORD = "Neer@j080105"
NEO4J_DATABASE = "copilot-kg-v6"


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


# ============================================================
# CONSTRAINTS
# ============================================================

CONSTRAINTS = [

    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT super_topic_name IF NOT EXISTS FOR (s:SuperTopic) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT persona_name IF NOT EXISTS FOR (p:Persona) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT intent_name IF NOT EXISTS FOR (i:Intent) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT product_name IF NOT EXISTS FOR (pr:Product) REQUIRE pr.name IS UNIQUE",
    "CREATE CONSTRAINT document_url IF NOT EXISTS FOR (d:Document) REQUIRE d.url IS UNIQUE"

]


# ============================================================
# FETCH DATA FROM POSTGRES
# ============================================================

def fetch_rows(source_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            ec.canonical_topic,
            ec.super_topic,
            ec.persona,
            ec.intent,
            d.url,
            d.title,
            d.product
        FROM ingestion.enriched_chunks ec
        JOIN ingestion.chunks c
            ON ec.chunk_id = c.id
        JOIN ingestion.documents d
            ON d.id = c.document_id
        WHERE ec.enrichment_version = %s
        AND d.source_id = %s
    """, (ENRICHMENT_VERSION, source_id))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


# ============================================================
# GRAPH UPSERT
# ============================================================

UPSERT_GRAPH = """
MERGE (t:Topic {name:$topic})

MERGE (s:SuperTopic {name:$super_topic})
MERGE (s)-[:HAS_TOPIC]->(t)

MERGE (p:Persona {name:$persona})
MERGE (t)-[:TARGETS_PERSONA]->(p)

MERGE (i:Intent {name:$intent})
MERGE (t)-[:SUPPORTS_INTENT]->(i)

MERGE (pr:Product {name:$product})
MERGE (t)-[:USED_IN_PRODUCT]->(pr)

MERGE (d:Document {url:$url})
SET d.title=$title

MERGE (d)-[:COVERS_TOPIC]->(t)
"""


# ============================================================
# BUILD RELATED TOPICS
# ============================================================

RELATED_TOPICS_QUERY = """
MATCH (t1:Topic)-[:HAS_TOPIC]-(s:SuperTopic)-[:HAS_TOPIC]-(t2:Topic)
WHERE t1 <> t2
MERGE (t1)-[:RELATED_TO]->(t2)
"""


# ============================================================
# MAIN
# ============================================================

def main(source_id):

    rows = fetch_rows(source_id)

    print(f"\n🧠 Building KG v6 from {len(rows)} enriched chunks\n")

    with driver.session(database=NEO4J_DATABASE) as session:

        # create constraints
        for c in CONSTRAINTS:
            session.run(c)

        # insert nodes
        for row in rows:

            topic, super_topic, persona, intent, url, title, product = row

            session.run(
                UPSERT_GRAPH,
                topic=topic or "Unknown",
                super_topic=super_topic or "General",
                persona=persona or "ProDeveloper",
                intent=intent or "reference",
                url=url,
                title=title or "",
                product=product or "Copilot"
            )

    print("🎉 Neo4j KG ingestion complete\n")

if __name__ == "__main__":
    import sys
    
if len(sys.argv) < 2:
        print("Usage: python -m ingestion.ingest_kg <source_id>")
else:
        main(sys.argv[1])
        print("✔ Ontology nodes inserted")

        # build topic relations
        session.run(RELATED_TOPICS_QUERY)

        print("✔ Topic relationships created")

        print("\n🎉 KG v6 build complete\n")

        import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_kg_v6.py <source_id>")
        sys.exit(1)

    source_id = sys.argv[1]
    main(source_id)
