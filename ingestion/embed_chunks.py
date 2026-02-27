# ingestion/embed_chunks.py

import time
import os
from typing import List
from ingestion.db import get_connection
from openai import OpenAI

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

# ============================================================
# 🔹 CONFIG
# ============================================================

ENRICHMENT_VERSION = 1
EMBEDDING_VERSION = 1

MODEL_NAME = "text-embedding-3-large"
VECTOR_DIMENSION = 3072

DEFAULT_COLLECTION = "project_chunks_v5"

BATCH_SIZE = 32
LLM_SLEEP = 0.02

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# 🔹 MILVUS COLLECTION
# ============================================================

def get_or_create_collection(collection_name):

    if utility.has_collection(collection_name):
        print("📦 Loading existing collection...")
        collection = Collection(collection_name)
        collection.load()
        return collection

    print(f"🆕 Creating collection: {collection_name}")

    fields = [
        FieldSchema("chunk_id", DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION),

        FieldSchema("product", DataType.VARCHAR, max_length=128),
        FieldSchema("topic", DataType.VARCHAR, max_length=256),
        FieldSchema("canonical_topic", DataType.VARCHAR, max_length=256),
        FieldSchema("super_topic", DataType.VARCHAR, max_length=256),
        FieldSchema("chunk_type", DataType.VARCHAR, max_length=64),
        FieldSchema("source_url", DataType.VARCHAR, max_length=2048),
        FieldSchema("document_title", DataType.VARCHAR, max_length=1024),
        FieldSchema("persona", DataType.VARCHAR, max_length=128),
        FieldSchema("intent", DataType.VARCHAR, max_length=64),
        FieldSchema("complexity", DataType.VARCHAR, max_length=64),
        FieldSchema("confidence", DataType.FLOAT),
        FieldSchema("text", DataType.VARCHAR, max_length=65535),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Unified KG-RAG chunks",
        enable_dynamic_field=False,
    )

    collection = Collection(collection_name, schema)

    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 32, "efConstruction": 200},
        },
    )

    collection.load()
    return collection


# ============================================================
# 🔹 FETCH CHUNKS TO EMBED
# ============================================================

def fetch_chunks_to_embed(source_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            c.id,
            c.chunk_text,
            c.topic,
            c.chunk_type,
            c.section_title,
            d.product,
            d.url,
            d.title,
            ec.persona,
            ec.intent,
            ec.complexity,
            ec.confidence,
            ec.canonical_topic,
            ec.super_topic
        FROM ingestion.chunks c
        JOIN ingestion.documents d 
            ON d.id = c.document_id
        JOIN ingestion.enriched_chunks ec 
            ON ec.chunk_id = c.id
            AND ec.enrichment_version = %s
        WHERE d.source_id = %s
        AND NOT EXISTS (
            SELECT 1
            FROM ingestion.embeddings e
            WHERE e.chunk_id = c.id
            AND e.embedding_version = %s
        )
    """, (
        ENRICHMENT_VERSION,
        source_id,
        EMBEDDING_VERSION
    ))

    rows = cur.fetchall()
    conn.close()
    return rows


# ============================================================
# 🔹 EMBEDDING
# ============================================================

def embed(text: str) -> List[float]:
    res = client.embeddings.create(
        model=MODEL_NAME,
        input=text[:8000]
    )
    return res.data[0].embedding


# ============================================================
# 🔹 AUDIT TABLE INSERT
# ============================================================

def insert_embedding_audit_batch(records, collection_name):
    if not records:
        return

    conn = get_connection()
    cur = conn.cursor()

    cur.executemany("""
        INSERT INTO ingestion.embeddings (
            chunk_id,
            embedding_version,
            embedding_model,
            vector_dimension,
            milvus_collection,
            embedded_at,
            ingestion_run_id
        )
        VALUES (%s,%s,%s,%s,%s,NOW(),NULL)
        ON CONFLICT DO NOTHING
    """, [
        (
            chunk_id,
            EMBEDDING_VERSION,
            MODEL_NAME,
            VECTOR_DIMENSION,
            collection_name,
        )
        for chunk_id in records
    ])

    conn.commit()
    conn.close()


# ============================================================
# 🔹 MAIN
# ============================================================

def main(source_id, collection_name=DEFAULT_COLLECTION):

    # Connect to Milvus
    connections.connect(alias="default", host="localhost", port="19530")

    collection = get_or_create_collection(collection_name)
    rows = fetch_chunks_to_embed(source_id)

    print(f"🧠 Embedding {len(rows)} chunks for source: {source_id}\n")

    total = 0
    batch = []
    audit_batch = []

    for row in rows:

        (
            chunk_id,
            text,
            topic,
            chunk_type,
            section_title,
            product,
            source_url,
            document_title,
            persona,
            intent,
            complexity,
            confidence,
            canonical_topic,
            super_topic
        ) = row

        vector = embed(text)

        batch.append({
            "chunk_id": str(chunk_id),
            "embedding": vector,
            "product": product,
            "topic": topic,
            "canonical_topic": canonical_topic,
            "super_topic": super_topic,
            "chunk_type": chunk_type,
            "source_url": source_url,
            "document_title": document_title,
            "persona": persona,
            "intent": intent,
            "complexity": complexity,
            "confidence": confidence,
            "text": text
        })

        audit_batch.append(chunk_id)

        if len(batch) >= BATCH_SIZE:
            collection.insert(batch)
            insert_embedding_audit_batch(audit_batch, collection_name)
            batch.clear()
            audit_batch.clear()

        total += 1
        time.sleep(LLM_SLEEP)

    if batch:
        collection.insert(batch)
        insert_embedding_audit_batch(audit_batch, collection_name)

    collection.flush()

    print("🎉 Embedding complete")
    print(f"🧩 Chunks embedded: {total}")