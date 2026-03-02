from pymilvus import connections, Collection

connections.connect(host="localhost", port="29530")

collection = Collection("copilot_studio_docs")
collection.load()

print("Total entities:", collection.num_entities)

# Search ONLY lab URLs
results = collection.query(
    expr='url like "%labs%"',
    output_fields=["url", "content"],
    limit=10
)

if not results:
    print("❌ NO LAB DATA FOUND IN DB")
else:
    print("\n✅ LAB DATA FOUND:\n")
    for r in results:
        print("URL:", r["url"])
        print(r["content"][:400])
        print("-"*60)