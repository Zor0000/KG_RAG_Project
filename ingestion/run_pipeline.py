"""
run_pipeline_all_products.py
────────────────────────────
Run this file to crawl, clean, chunk, enrich, embed, and ingest ALL
Microsoft product sources into Milvus + Neo4j.

Previously only copilot_studio was indexed — that is why queries about
labs for M365, Power Platform, Azure AI, and business users returned
weak or empty results.

Run each product one at a time:
    python run_pipeline_all_products.py --product m365
    python run_pipeline_all_products.py --product power_platform
    python run_pipeline_all_products.py --product azure_ai
    python run_pipeline_all_products.py --product copilot_studio   (re-run to fill lab gaps)
    python run_pipeline_all_products.py --product all              (run everything)
"""

import asyncio
import time
import argparse

from ingestion.crawl_docs    import main as crawl_main
from ingestion.clean_markdown import main as clean_main
from ingestion.chunk_docs    import main as chunk_main
from ingestion.enrich_chunks import main as enrich_main
from ingestion.embed_chunks  import main as embed_main
from ingestion.ingest_kg     import main as kg_main

MILVUS_COLLECTION = "project_chunks_v5"


# ============================================================
# 🔹 PRODUCT CONFIGS
# Each product has:
#   - start_urls  : entry points for the crawler
#   - allowed_prefixes : crawler stays within these paths
#   - lab_urls    : specific lab/exercise pages (crawled with higher priority)
#                   These fill the gap for "which labs are relevant" queries.
# ============================================================

PRODUCT_CONFIGS = {

    # ── Copilot Studio (existing — extended with lab URLs) ────────────
    "copilot_studio": {
        "source_id": "copilot_studio",
        "start_urls": [
            "https://learn.microsoft.com/en-us/microsoft-copilot-studio/",
            "https://adoption.microsoft.com/en-us/ai-agents/copilot-studio/",
            # Official lab site — THIS is where lab recommendation content lives
            "https://pratapladhani.github.io/mcs-labs/",
            "https://learn.microsoft.com/en-us/training/browse/?products=power-virtual-agents",
            "https://learn.microsoft.com/en-us/training/paths/work-power-virtual-agents/",
        ],
        "allowed_prefixes": [
            "https://learn.microsoft.com/en-us/microsoft-copilot-studio/",
            "https://adoption.microsoft.com/en-us/ai-agents/copilot-studio/",
            "https://pratapladhani.github.io/mcs-labs/",
            "https://learn.microsoft.com/en-us/training/",
        ],
    },

    # ── Microsoft 365 ─────────────────────────────────────────────────
    # Covers Word, Excel, Teams, Outlook, SharePoint, M365 Copilot
    "m365": {
        "source_id": "m365",
        "start_urls": [
            "https://learn.microsoft.com/en-us/microsoft-365/",
            "https://learn.microsoft.com/en-us/microsoftteams/",
            "https://learn.microsoft.com/en-us/training/m365/",
            "https://adoption.microsoft.com/en-us/copilot/",
            # M365 Copilot lab/scenario pages
            "https://learn.microsoft.com/en-us/training/paths/explore-microsoft-365-copilot/",
            "https://learn.microsoft.com/en-us/training/paths/get-started-with-microsoft-365-copilot/",
        ],
        "allowed_prefixes": [
            "https://learn.microsoft.com/en-us/microsoft-365/",
            "https://learn.microsoft.com/en-us/microsoftteams/",
            "https://learn.microsoft.com/en-us/training/m365/",
            "https://learn.microsoft.com/en-us/training/paths/",
            "https://adoption.microsoft.com/en-us/copilot/",
        ],
    },

    # ── Power Platform ────────────────────────────────────────────────
    # Covers Power Apps, Power Automate, Power BI, Power Pages
    "power_platform": {
        "source_id": "power_platform",
        "start_urls": [
            "https://learn.microsoft.com/en-us/power-platform/",
            "https://learn.microsoft.com/en-us/power-apps/",
            "https://learn.microsoft.com/en-us/power-automate/",
            "https://learn.microsoft.com/en-us/power-bi/",
            # Power Platform learning paths and labs
            "https://learn.microsoft.com/en-us/training/powerplatform/",
            "https://learn.microsoft.com/en-us/training/paths/create-powerapps/",
            "https://learn.microsoft.com/en-us/training/paths/automate-process-power-automate/",
        ],
        "allowed_prefixes": [
            "https://learn.microsoft.com/en-us/power-platform/",
            "https://learn.microsoft.com/en-us/power-apps/",
            "https://learn.microsoft.com/en-us/power-automate/",
            "https://learn.microsoft.com/en-us/power-bi/",
            "https://learn.microsoft.com/en-us/training/powerplatform/",
            "https://learn.microsoft.com/en-us/training/paths/",
        ],
    },

    # ── Azure AI / Azure OpenAI ───────────────────────────────────────
    "azure_ai": {
        "source_id": "azure_ai",
        "start_urls": [
            "https://learn.microsoft.com/en-us/azure/ai-services/",
            "https://learn.microsoft.com/en-us/azure/ai-studio/",
            "https://learn.microsoft.com/en-us/azure/cognitive-services/openai/",
            # Azure AI learning paths and labs
            "https://learn.microsoft.com/en-us/training/paths/develop-ai-solutions-azure/",
            "https://learn.microsoft.com/en-us/training/paths/azure-ai-fundamentals/",
            "https://microsoftlearning.github.io/AI-102-AIEngineer/",
        ],
        "allowed_prefixes": [
            "https://learn.microsoft.com/en-us/azure/ai-services/",
            "https://learn.microsoft.com/en-us/azure/ai-studio/",
            "https://learn.microsoft.com/en-us/azure/cognitive-services/openai/",
            "https://learn.microsoft.com/en-us/training/paths/",
            "https://microsoftlearning.github.io/AI-102-AIEngineer/",
        ],
    },
}


# ============================================================
# 🔹 SINGLE PRODUCT PIPELINE
# ============================================================

async def run_product(product_key: str):
    cfg = PRODUCT_CONFIGS[product_key]
    source_id = cfg["source_id"]

    print("\n" + "=" * 60)
    print(f"🚀 PIPELINE: {product_key.upper()}")
    print("=" * 60 + "\n")

    # 1. Crawl
    print("🌐 STEP 1: Crawling")
    run_id = await crawl_main(
        product         = product_key,
        source_id       = source_id,
        start_urls      = cfg["start_urls"],
        allowed_prefixes= cfg["allowed_prefixes"],
        resume          = True,   # skip already-crawled URLs automatically
    )
    print(f"✅ Crawl complete | Run ID: {run_id}\n")
    time.sleep(1)

    # 2. Clean
    print("🧹 STEP 2: Cleaning")
    clean_main(source_id=source_id, run_id=run_id)
    print("✅ Cleaning complete\n")
    time.sleep(1)

    # 3. Chunk
    print("🧩 STEP 3: Chunking")
    chunk_main(source_id=source_id, run_id=run_id)
    print("✅ Chunking complete\n")
    time.sleep(1)

    # 4. Enrich
    print("🧠 STEP 4: Enriching")
    enrich_main(source_id=source_id, run_id=run_id)
    print("✅ Enrichment complete\n")
    time.sleep(1)

    # 5. Embed
    print("📐 STEP 5: Embedding")
    embed_main(source_id=source_id, collection_name=MILVUS_COLLECTION)
    print("✅ Embedding complete\n")
    time.sleep(1)

    # 6. KG ingest
    print("🧠 STEP 6: KG Ingestion")
    kg_main(source_id=source_id)
    print("✅ KG ingestion complete\n")

    print("=" * 60)
    print(f"🎉 DONE: {product_key.upper()}")
    print("=" * 60 + "\n")


# ============================================================
# 🔹 ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--product",
        choices=list(PRODUCT_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which product to crawl and index"
    )
    args = parser.parse_args()

    if args.product == "all":
        for pk in PRODUCT_CONFIGS:
            asyncio.run(run_product(pk))
    else:
        asyncio.run(run_product(args.product))