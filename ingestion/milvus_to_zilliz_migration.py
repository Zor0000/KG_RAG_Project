#pip install pymilvus
from pymilvus import connections, Collection, utility
import math

# =============================
# CONFIGURATION
# =============================

LOCAL_HOST = "localhost"
LOCAL_PORT = "19530"

ZILLIZ_URI = "https://in03-7f4e2db6c0219db.serverless.aws-eu-central-1.cloud.zilliz.com"
ZILLIZ_TOKEN = "09c817cbd71c4f61f3c892ba66906a7af9aeb16df92ecdc4c024c7c1c5cfe75e82df48f3899cb7934a56c01f2d069c31f9f0147e"

COLLECTION_NAME = "project_chunks_v5"

BATCH_SIZE = 1000


# =============================
# CONNECT TO DATABASES
# =============================

print("Connecting to local Milvus...")
connections.connect(
    alias="local",
    host=LOCAL_HOST,
    port=LOCAL_PORT
)

print("Connecting to Zilliz Cloud...")
connections.connect(
    alias="cloud",
    uri=ZILLIZ_URI,
    token=ZILLIZ_TOKEN
)

# =============================
# LOAD LOCAL COLLECTION
# =============================

local_collection = Collection(
    COLLECTION_NAME,
    using="local"
)

schema = local_collection.schema
print("Schema loaded")

# =============================
# CREATE COLLECTION IN CLOUD
# =============================

if utility.has_collection(COLLECTION_NAME, using="cloud"):
    print("Collection already exists in cloud")
    cloud_collection = Collection(COLLECTION_NAME, using="cloud")
else:
    print("Creating collection in cloud")
    cloud_collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using="cloud"
    )

# =============================
# LOAD LOCAL DATA
# =============================

local_collection.load()

total = local_collection.num_entities
print(f"Total entities: {total}")

num_batches = math.ceil(total / BATCH_SIZE)

fields = [f.name for f in schema.fields]

# =============================
# DATA MIGRATION
# =============================

print("Starting migration...")

for i in range(num_batches):

    offset = i * BATCH_SIZE

    print(f"Migrating batch {i+1}/{num_batches}")

    data = local_collection.query(
        expr="",
        output_fields=fields,
        offset=offset,
        limit=BATCH_SIZE
    )

    if data:
        cloud_collection.insert(data)

cloud_collection.flush()

print("Data migration completed")

# =============================
# REBUILD INDEX
# =============================

vector_field = None

for f in schema.fields:
    if f.dtype.name == "FLOAT_VECTOR":
        vector_field = f.name

if vector_field:
    print("Creating index on vector field")

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200}
    }

    cloud_collection.create_index(
        field_name=vector_field,
        index_params=index_params
    )

cloud_collection.load()

print("Index created")

# =============================
# VERIFY MIGRATION
# =============================

print("Verifying...")

print("Local count:", total)
print("Cloud count:", cloud_collection.num_entities)

print("Migration finished successfully!")