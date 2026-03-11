# ingestion/embed_chunks.py

import time
import os
import random
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
EMBEDDING_VERSION  = 1

MODEL_NAME       = "text-embedding-3-large"
VECTOR_DIMENSION = 3072

DEFAULT_COLLECTION = "project_chunks_v5"

BATCH_SIZE = 32
LLM_SLEEP  = 0.02

# Retry config
MAX_RETRIES     = 5
RETRY_BASE_WAIT = 10   # seconds, doubles each attempt

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# 🔹 RETRY WRAPPER
# ============================================================

def call_with_retry(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e).lower()
            is_retryable = (
                "rate" in err or "429" in err or "quota" in err
                or "timeout" in err or "timed out" in err
                or "500" in err or "503" in err or "server" in err
                or "connection" in err
            )

            if attempt == MAX_RETRIES:
                print(f"  ⚠️  Giving up after {MAX_RETRIES} attempts: {e}")
                return None

            if is_retryable:
                wait = RETRY_BASE_WAIT * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"  🔄 Attempt {attempt}/{MAX_RETRIES} failed ({type(e).__name__}). "
                      f"Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Non-retryable error: {e}")
                return None

    return None


# ============================================================
# 🔹 MILVUS COLLECTION
# ============================================================

def get_or_create_collection(collection_name):

    if utility.has_collection(collection_name):
        print("📦 Loading existing Milvus collection...")
        collection = Collection(collection_name)
        collection.load()
        return collection

    print(f"🆕 Creating new Milvus collection: {collection_name}")

    fields = [
        FieldSchema("chunk_id",       DataType.VARCHAR,      max_length=128,   is_primary=True),
        FieldSchema("embedding",      DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION),
        FieldSchema("product",        DataType.VARCHAR,      max_length=128),
        FieldSchema("topic",          DataType.VARCHAR,      max_length=256),
        FieldSchema("canonical_topic",DataType.VARCHAR,      max_length=256),
        FieldSchema("super_topic",    DataType.VARCHAR,      max_length=256),
        FieldSchema("chunk_type",     DataType.VARCHAR,      max_length=64),
        FieldSchema("source_url",     DataType.VARCHAR,      max_length=2048),
        FieldSchema("document_title", DataType.VARCHAR,      max_length=1024),
        FieldSchema("persona",        DataType.VARCHAR,      max_length=128),
        FieldSchema("intent",         DataType.VARCHAR,      max_length=64),
        FieldSchema("complexity",     DataType.VARCHAR,      max_length=64),
        FieldSchema("confidence",     DataType.FLOAT),
        FieldSchema("text",           DataType.VARCHAR,      max_length=65535),
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
    """
    Only returns chunks NOT already in ingestion.embeddings.
    Re-running after a crash automatically resumes from where it stopped.
    """
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
        SELECT
            c.id, c.chunk_text, c.topic, c.chunk_type, c.section_title,
            d.product, d.url, d.title,
            ec.persona, ec.intent, ec.complexity, ec.confidence,
            ec.canonical_topic, ec.super_topic
        FROM ingestion.chunks c
        JOIN ingestion.documents d    ON d.id = c.document_id
        JOIN ingestion.enriched_chunks ec
            ON ec.chunk_id = c.id
            AND ec.enrichment_version = %s
        WHERE d.source_id = %s
        AND NOT EXISTS (
            SELECT 1 FROM ingestion.embeddings e
            WHERE e.chunk_id = c.id
            AND e.embedding_version = %s
        )
        ORDER BY c.id
    """, (ENRICHMENT_VERSION, source_id, EMBEDDING_VERSION))

    rows = cur.fetchall()
    conn.close()
    return rows


def get_already_embedded_count(source_id):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM ingestion.embeddings e
        JOIN ingestion.chunks c     ON c.id = e.chunk_id
        JOIN ingestion.documents d  ON d.id = c.document_id
        WHERE d.source_id = %s AND e.embedding_version = %s
    """, (source_id, EMBEDDING_VERSION))
    count = cur.fetchone()[0]
    conn.close()
    return count


# ============================================================
# 🔹 EMBEDDING
# ============================================================

def embed(text: str) -> List[float]:
    def _call():
        res = client.embeddings.create(
            model=MODEL_NAME,
            input=text[:8000]
        )
        return res.data[0].embedding

    return call_with_retry(_call)


# ============================================================
# 🔹 SAFE STRING HELPER
# ============================================================

def safe_str(val, max_len: int, default: str = "") -> str:
    """Truncate and sanitise strings before inserting into Milvus."""
    if val is None:
        return default
    return str(val)[:max_len]


# ============================================================
# 🔹 AUDIT TABLE INSERT
# ============================================================

def insert_embedding_audit_batch(records, collection_name):
    if not records:
        return

    conn = get_connection()
    cur  = conn.cursor()

    cur.executemany("""
        INSERT INTO ingestion.embeddings (
            chunk_id, embedding_version, embedding_model,
            vector_dimension, milvus_collection, embedded_at, ingestion_run_id
        )
        VALUES (%s,%s,%s,%s,%s,NOW(),NULL)
        ON CONFLICT DO NOTHING
    """, [
        (chunk_id, EMBEDDING_VERSION, MODEL_NAME, VECTOR_DIMENSION, collection_name)
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

    rows          = fetch_chunks_to_embed(source_id)
    total_remaining = len(rows)
    already_done  = get_already_embedded_count(source_id)
    total_all     = already_done + total_remaining

    print(f"📐 Embedding status for: {source_id}")
    print(f"   ✅ Already done : {already_done}/{total_all}")
    print(f"   ⏳ Remaining    : {total_remaining}/{total_all}")
    print(f"   ⚡ Est. time    : ~{int(total_remaining * 0.05 / 60)} mins\n")

    if total_remaining == 0:
        print("✅ All chunks already embedded — nothing to do.")
        return

    total       = 0
    skipped     = 0
    batch       = []
    audit_batch = []

    try:
        for row in rows:

            (
                chunk_id, text, topic, chunk_type, section_title,
                product, source_url, document_title,
                persona, intent, complexity, confidence,
                canonical_topic, super_topic
            ) = row

            # Get embedding — retry handles rate limits
            vector = embed(text)

            if vector is None:
                print(f"  ⚠️  Skipping chunk {chunk_id} — embedding failed after retries")
                skipped += 1
                continue

            batch.append({
                "chunk_id":       safe_str(chunk_id,       128),
                "embedding":      vector,
                "product":        safe_str(product,        128,   "unknown"),
                "topic":          safe_str(topic,          256,   ""),
                "canonical_topic":safe_str(canonical_topic,256,   ""),
                "super_topic":    safe_str(super_topic,    256,   "Agent Fundamentals"),
                "chunk_type":     safe_str(chunk_type,     64,    "text"),
                "source_url":     safe_str(source_url,     2048,  ""),
                "document_title": safe_str(document_title, 1024,  ""),
                "persona":        safe_str(persona,        128,   "ProDeveloper"),
                "intent":         safe_str(intent,         64,    "reference"),
                "complexity":     safe_str(complexity,     64,    "pro-code"),
                "confidence":     float(confidence) if confidence else 0.5,
                "text":           safe_str(text,           65535, ""),
            })

            audit_batch.append(chunk_id)

            if len(batch) >= BATCH_SIZE:
                collection.insert(batch)
                insert_embedding_audit_batch(audit_batch, collection_name)
                batch.clear()
                audit_batch.clear()

            total += 1

            if total % 100 == 0:
                pct = (total / total_remaining) * 100
                print(f"  📊 {total}/{total_remaining} remaining ({pct:.1f}%) "
                      f"— total done: {already_done + total}/{total_all}")

            time.sleep(LLM_SLEEP)

        # Flush remaining batch
        if batch:
            collection.insert(batch)
            insert_embedding_audit_batch(audit_batch, collection_name)

        collection.flush()

        print("\n🎉 Embedding complete")
        print(f"🧩 Chunks embedded : {total}")
        if skipped:
            print(f"⚠️  Chunks skipped  : {skipped} (embedding API failed)")

    except Exception as e:
        # Flush whatever is in the batch before raising
        if batch:
            try:
                collection.insert(batch)
                insert_embedding_audit_batch(audit_batch, collection_name)
                collection.flush()
                print(f"  💾 Saved {len(batch)} chunks before crash")
            except Exception:
                pass
        print(f"\n💥 Embedding failed after {total} chunks: {e}")
        raise


# ============================================================
# 🔹 ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else "copilot_studio"
    main(source_id=sid)