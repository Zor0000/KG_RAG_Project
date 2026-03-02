import os
import json
import sys
from pathlib import Path

# Fix local imports
sys.path.append("retrieval")

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

from sentence_transformers import SentenceTransformer

# Topic chunking (your custom chunking)
from topic_chunking import topic_based_chunking


# =============================
# CONFIG
# =============================

DATA_DIR = Path("data/raw_pages")
COLLECTION_NAME = "copilot_studio_docs"

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "29530"   # Docker mapped port


# =============================
# CONNECT MILVUS
# =============================

print("🔌 Connecting to Milvus...")

connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

print("✅ Milvus connected")


# =============================
# LOAD EMBEDDING MODEL
# =============================

print("⚡ Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# =============================
# COLLECTION SETUP
# =============================

def create_collection():
    if utility.has_collection(COLLECTION_NAME):
        print("✅ Collection exists")
        collection = Collection(COLLECTION_NAME)
        collection.load()
        return collection

    print("⚡ Creating new collection...")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=384,
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65000,
        ),
        FieldSchema(
            name="url",
            dtype=DataType.VARCHAR,
            max_length=2000,
        ),
    ]

    schema = CollectionSchema(fields)

    collection = Collection(COLLECTION_NAME, schema)

    # Create vector index
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        },
    )

    collection.load()
    print("✅ Collection created")

    return collection


# =============================
# INSERT DATA
# =============================

def insert_embeddings():
    collection = create_collection()

    files = list(DATA_DIR.glob("*.json"))

    print(f"📄 Found {len(files)} pages")

    all_embeddings = []
    all_contents = []
    all_urls = []

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        url = data.get("url", "")
        text = data.get("content", "")

        # ⭐ Topic-based chunking (your upgrade)
        chunks = topic_based_chunking(text)

        for chunk in chunks:
            emb = model.encode(chunk).tolist()

            all_embeddings.append(emb)
            all_contents.append(chunk)
            all_urls.append(url)

    print(f"🚀 Inserting {len(all_embeddings)} chunks...")

    # IMPORTANT → Milvus expects column format
    collection.insert([
        all_embeddings,
        all_contents,
        all_urls,
    ])

    collection.flush()
    collection.load()

    print("🎉 DONE — Data inserted into Milvus")


if __name__ == "__main__":
    insert_embeddings()
