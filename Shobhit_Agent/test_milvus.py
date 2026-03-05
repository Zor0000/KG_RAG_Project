from pymilvus import connections

connections.connect(
    host="localhost",
    port="29530"
)

print("Milvus connected successfully!")
