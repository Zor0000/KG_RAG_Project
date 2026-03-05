from pymilvus import Collection
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
collection = Collection("copilot_docs")

def vector_search(query: str, top_k=30):
    embedding = model.encode([query])[0]

    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )

    return [hit.entity.get("text") for hit in results[0]]
