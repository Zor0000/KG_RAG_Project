from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

connections.connect(host="localhost", port="29530")

collection = Collection("copilot_studio_docs")
collection.load()

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "How to create copilot agent in Copilot Studio?"
embedding = model.encode(query).tolist()

results = collection.search(
    data=[embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=5,
    output_fields=["content", "url"]
)

for hit in results[0]:
    print("\nScore:", hit.score)
    print(hit.entity.get("url"))
    print(hit.entity.get("content")[:200])
