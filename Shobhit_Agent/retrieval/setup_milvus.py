from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

connections.connect(host="localhost", port="29530")

COLLECTION_NAME = "copilot_studio_docs"

def create_collection():

    if utility.has_collection(COLLECTION_NAME):
        print("Collection exists")
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64,
                    is_primary=True, auto_id=True),

        FieldSchema(name="embedding",
                    dtype=DataType.FLOAT_VECTOR, dim=384),

        FieldSchema(name="text",
                    dtype=DataType.VARCHAR, max_length=65535),

        FieldSchema(name="url",
                    dtype=DataType.VARCHAR, max_length=1024),

        FieldSchema(name="title",
                    dtype=DataType.VARCHAR, max_length=512)
    ]

    schema = CollectionSchema(fields)
    Collection(COLLECTION_NAME, schema)

    print("Milvus collection ready")

if __name__ == "__main__":
    create_collection()
