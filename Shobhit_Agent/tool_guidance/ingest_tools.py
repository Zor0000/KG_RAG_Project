import json
import re
from pathlib import Path
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent

MILVUS_HOST = "localhost"
MILVUS_PORT = "29530"
COLLECTION_NAME = "copilot_studio_docs"

model = SentenceTransformer("all-MiniLM-L6-v2")


def normalize_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\xa0', ' ')
    return text.strip()


def connect_milvus():
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )

    collection = Collection(COLLECTION_NAME)
    collection.load()

    print("✅ Milvus connected")
    return collection


def chunk_text(text, size=400):
    text = normalize_text(text)
    words = text.split()

    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]


def ingest_json(file_path):
    collection = connect_milvus()

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    count = 0

    for item in data:
        content = normalize_text(item.get("content", ""))
        url = item.get("url", "")

        if not content:
            continue

        chunks = chunk_text(content)

        for chunk in chunks:
            emb = model.encode(chunk).tolist()

            collection.insert([
                [emb],
                [chunk],
                [url]
            ])

            count += 1

    print(f"✅ Inserted {count} chunks safely")


if __name__ == "__main__":
    ingest_json(BASE / "official_labs.json")