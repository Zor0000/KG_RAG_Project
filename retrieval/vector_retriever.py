# retrieval/vector_retriever.py

from typing import List, Dict
from pymilvus import connections, Collection, utility
from openai import OpenAI
import os


# ============================================================
# 🔹 CONFIG
# ============================================================

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

COLLECTION_NAME = "project_chunks_v5"
VECTOR_FIELD = "embedding"

MODEL_NAME = "text-embedding-3-large"
TOP_K = 30

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================
# 🔹 Initialize
# ============================================================

print("🔌 Connecting to Milvus...")
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# Check if collection exists
if utility.has_collection(COLLECTION_NAME):
    collection = Collection(COLLECTION_NAME)
    collection.load()
    print(f"✅ Collection {COLLECTION_NAME} loaded successfully")
else:
    raise Exception(
        f"❌ Collection '{COLLECTION_NAME}' does not exist in Milvus. Please run ingestion first."
    )

print("🤖 Initializing OpenAI embedding client...")
client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# 🔹 Generate Query Embedding
# ============================================================

def embed_query(query: str) -> List[float]:

    response = client.embeddings.create(
        model=MODEL_NAME,
        input=query
    )

    return response.data[0].embedding


# ============================================================
# 🔹 Vector Search
# ============================================================

def vector_search(
    query: str,
    top_k: int = TOP_K,
    persona_filter: str = None,
    product_filter: str = None
) -> List[Dict]:

    query_vector = embed_query(query)

    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }

    expr = []

    if persona_filter and persona_filter.lower() != "all":
        expr.append(f'persona == "{persona_filter}"')

    if product_filter and product_filter.lower() != "all":
        expr.append(f'product == "{product_filter}"')

    filter_expr = " && ".join(expr) if expr else None

    results = collection.search(
        data=[query_vector],
        anns_field=VECTOR_FIELD,
        param=search_params,
        limit=top_k,
        expr=filter_expr,
        output_fields=[
            "chunk_id",
            "text",
            "persona",
            "intent",
            "canonical_topic",
            "super_topic",
            "product",
            "source_url"
        ]
    )

    hits = results[0]

    formatted_results = []

    for hit in hits:
        formatted_results.append({
            "score": float(hit.score),
            "chunk_id": hit.entity.get("chunk_id"),
            "text": hit.entity.get("text"),
            "persona": hit.entity.get("persona"),
            "intent": hit.entity.get("intent"),
            "canonical_topic": hit.entity.get("canonical_topic"),
            "super_topic": hit.entity.get("super_topic"),
            "product": hit.entity.get("product"),
            "source_url": hit.entity.get("source_url"),
        })

    return formatted_results


# ============================================================
# 🔹 CLI Test
# ============================================================

if __name__ == "__main__":

    while True:
        query = input("\n💬 Enter query (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = vector_search(query)

        print(f"\n🔎 Top {len(results)} Results:\n")

        for i, r in enumerate(results, start=1):
            print(f"{i}. Score: {r['score']:.4f}")
            print(f"   Persona: {r['persona']}")
            print(f"   Intent: {r['intent']}")
            print(f"   Topic: {r['canonical_topic']}")
            print(f"   URL: {r['source_url']}")
            print("   Text Preview:", r["text"][:200], "...\n")