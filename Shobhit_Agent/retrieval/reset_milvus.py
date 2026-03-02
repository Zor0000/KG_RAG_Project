from pymilvus import connections, utility

connections.connect(host="localhost", port="29530")

COLLECTION_NAME = "copilot_studio_docs"

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
    print("✅ Old collection deleted")
else:
    print("No collection found")
