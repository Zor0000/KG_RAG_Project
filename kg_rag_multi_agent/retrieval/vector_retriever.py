# kg_rag_multi_agent/retrieval/vector_retriever.py
#
# ⚠️ DISABLED — Milvus has been removed. Migrating to pgvector (PostgreSQL).
# All code below is commented out pending pgvector replacement.

# def vector_search(query, product):
#
#     results = milvus_collection.search(
#         data=[query_embedding],
#         anns_field="embedding",
#         limit=5,
#         filter=f'product == "{product}"'
#     )
#
#     return results