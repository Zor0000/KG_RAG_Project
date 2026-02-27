import asyncio
import time

from ingestion.crawl_docs import main as crawl_main
from ingestion.clean_markdown import main as clean_main
from ingestion.chunk_docs import main as chunk_main
from ingestion.enrich_chunks import main as enrich_main
from ingestion.embed_chunks import main as embed_main
from ingestion.ingest_kg import main as kg_main


# ============================================================
# 🔹 PRODUCT CONFIG (EDIT THIS PER PRODUCT)
# ============================================================

PRODUCT = "autogen"

SOURCE_ID = "autogen-ecosystem"

START_URLS = [
    "https://microsoft.github.io/autogen/stable/",
]

ALLOWED_PREFIXES = [
    "https://microsoft.github.io/autogen/stable/",
]

MILVUS_COLLECTION = "project_chunks_v5"   # Keep same collection


# ============================================================
# 🔹 PIPELINE
# ============================================================

async def run_pipeline():

    print("\n" + "="*60)
    print(f"🚀 STARTING PIPELINE FOR: {PRODUCT}")
    print("="*60 + "\n")

    # --------------------------------------------------------
    # 1️⃣ CRAWL
    # --------------------------------------------------------
    print("🌐 STEP 1: Crawling")
    run_id = await crawl_main(
        product=PRODUCT,
        source_id=SOURCE_ID,
        start_urls=START_URLS,
        allowed_prefixes=ALLOWED_PREFIXES,
    )

    print(f"✅ Crawl complete | Run ID: {run_id}\n")
    time.sleep(1)

    # --------------------------------------------------------
    # 2️⃣ CLEAN
    # --------------------------------------------------------
    print("🧹 STEP 2: Cleaning")
    clean_main(
        source_id=SOURCE_ID,
        run_id=run_id
    )

    print("✅ Cleaning complete\n")
    time.sleep(1)

    # --------------------------------------------------------
    # 3️⃣ CHUNK
    # --------------------------------------------------------
    print("🧩 STEP 3: Chunking")
    chunk_main(
        source_id=SOURCE_ID,
        run_id=run_id
    )

    print("✅ Chunking complete\n")
    time.sleep(1)

    # --------------------------------------------------------
    # 4️⃣ ENRICH
    # --------------------------------------------------------
    print("🧠 STEP 4: Enrichment")
    enrich_main(
        source_id=SOURCE_ID,
        run_id=run_id
    )

    print("✅ Enrichment complete\n")
    time.sleep(1)

    # --------------------------------------------------------
    # 5️⃣ EMBED
    # --------------------------------------------------------
    print("📐 STEP 5: Embedding")
    embed_main(
        source_id=SOURCE_ID,
        collection_name=MILVUS_COLLECTION
    )

    print("✅ Embedding complete\n")
    time.sleep(1)

    # --------------------------------------------------------
    # 6️⃣ KG INGEST
    # --------------------------------------------------------
    print("🧠 STEP 6: Neo4j KG Ingestion")
    kg_main(
        source_id=SOURCE_ID
    )

    print("✅ KG ingestion complete\n")

    print("="*60)
    print("🎉 PIPELINE FINISHED SUCCESSFULLY")
    print("="*60 + "\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(run_pipeline())