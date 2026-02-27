import asyncio
import hashlib
import uuid
from collections import deque
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler
from ingestion.db import get_connection


# ============================================================
# 🔹 CONFIG
# ============================================================


DENY_PATTERNS = [
    "/resources/",
    "/related",
    "/toc",
    "/index",
    "/previous-versions/",
    "/migration/",
    "?tabs=",
    "?view=",
    "#",
]

MAX_PAGES = 15000

PIPELINE_VERSION = 1
CLEANING_VERSION = 1
CHUNKING_VERSION = 1
ENRICHMENT_VERSION = 1
EMBEDDING_VERSION = 1


# ============================================================
# 🔹 URL HELPERS
# ============================================================

def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    clean = parsed._replace(fragment="", query="").geturl()
    return clean.rstrip("/")


def extract_canonical_url(result, fallback_url: str) -> str:
    try:
        if hasattr(result, "metadata") and result.metadata:
            canonical = result.metadata.get("canonical")
            if canonical:
                return normalize_url(canonical)
    except Exception:
        pass

    return normalize_url(fallback_url)


def is_allowed(url: str, allowed_prefixes) -> bool:
    return any(url.startswith(prefix) for prefix in allowed_prefixes)


def is_denied(url: str) -> bool:
    return any(p in url for p in DENY_PATTERNS)


def is_low_content(markdown: str) -> bool:
    lines = [l for l in markdown.splitlines() if l.strip()]
    if len(lines) < 30:
        return True
    heading_ratio = sum(1 for l in lines if l.startswith("#")) / len(lines)
    return heading_ratio > 0.4


def extract_internal_links(base_url: str, result, allowed_prefixes) -> list[str]:
    urls = []

    if not result.links or "internal" not in result.links:
        return urls

    for link in result.links["internal"]:
        href = link.get("href")
        if not href:
            continue

        full = urljoin(base_url, href)
        clean = normalize_url(full)

        if not is_allowed(clean, allowed_prefixes):
            continue
        if is_denied(clean):
            continue

        urls.append(clean)

    return urls


# ============================================================
# 🔹 DB FUNCTIONS (Single Connection Version)
# ============================================================

def create_ingestion_run(cur, source_id):

    cur.execute("""
        INSERT INTO ingestion.ingestion_runs (
            id,
            source_id,
            pipeline_version,
            cleaning_version,
            chunking_version,
            enrichment_version,
            embedding_version,
            started_at,
            status
        )
        VALUES (
            gen_random_uuid(),
            %s,%s,%s,%s,%s,%s,
            NOW(),
            'running'
        )
        RETURNING id
    """, (
        source_id,
        PIPELINE_VERSION,
        CLEANING_VERSION,
        CHUNKING_VERSION,
        ENRICHMENT_VERSION,
        EMBEDDING_VERSION
    ))

    return str(cur.fetchone()[0])


def finish_ingestion_run(cur, run_id, status, total_docs):

    cur.execute("""
        UPDATE ingestion.ingestion_runs
        SET status = %s,
            finished_at = NOW(),
            total_documents = %s
        WHERE id = %s
    """, (status, total_docs, run_id))


def upsert_document(cur, source_id, product, url, canonical_url, markdown, depth):

    content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    lookup_url = canonical_url or url

    cur.execute("""
        SELECT id, content_hash
        FROM ingestion.documents
        WHERE source_id = %s AND url = %s
""", (source_id, lookup_url))

    result = cur.fetchone()

    if result:
        document_id, old_hash = result

        if old_hash == content_hash:
            return document_id, False

        cur.execute("""
            UPDATE ingestion.documents
            SET content_hash = %s,
                last_crawled_at = NOW(),
                version = version + 1
            WHERE id = %s
        """, (content_hash, document_id))

        return document_id, True

    else:
        document_id = str(uuid.uuid4())

        cur.execute("""
            INSERT INTO ingestion.documents (
                id,
                source_id,
                url,
                canonical_url,
                title,
                product,
                content_hash,
                http_status,
                crawl_depth,
                first_seen_at,
                last_crawled_at,
                version,
                is_active
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW(),1,TRUE)
        """, (
            document_id,
            source_id,
            lookup_url,
            lookup_url,
            "",
            product,
            content_hash,
            200,
            depth
        ))

        return document_id, True


def insert_raw_page(cur, document_id, markdown, run_id):

    raw_id = str(uuid.uuid4())
    content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

    cur.execute("""
        INSERT INTO ingestion.raw_pages (
            id,
            document_id,
            raw_content,
            content_hash,
            crawl_version,
            ingestion_run_id
        )
        VALUES (%s,%s,%s,%s,%s,%s)
    """, (
        raw_id,
        document_id,
        markdown,
        content_hash,
        1,
        run_id
    ))


# ============================================================
# 🔹 MAIN
# ============================================================

async def main(product, source_id, start_urls, allowed_prefixes):

    print("🚀 Starting standalone crawl")

    conn = get_connection()
    cur = conn.cursor()

    run_id = create_ingestion_run(cur, source_id)
    conn.commit()

    print(f"🧠 Ingestion Run ID: {run_id}")

    queue = deque([(url, 0) for url in start_urls])
    seen_in_memory = set(normalize_url(url) for url in start_urls)

    crawler = AsyncWebCrawler()

    processed_count = 0
    updated_documents = 0

    try:
        async with crawler:
            while queue and processed_count < MAX_PAGES:

                url, depth = queue.popleft()
                url = normalize_url(url)

                try:
                    result = await crawler.arun(url)
                except Exception as e:
                    print(f"❌ Crawl failed: {url} → {e}")
                    continue

                markdown = getattr(result, "markdown", None)
                if not markdown or len(markdown.strip()) < 200:
                    continue

                if is_low_content(markdown):
                    continue

                canonical_url = extract_canonical_url(result, url)

                document_id, changed = upsert_document(
                    cur,
                    source_id,
                    product,
                    url,
                    canonical_url,
                    markdown,
                    depth
                )

                if changed:
                    insert_raw_page(cur, document_id, markdown, run_id)
                    updated_documents += 1

                processed_count += 1

                # Commit every 50 pages (IMPORTANT)
                if processed_count % 50 == 0:
                    conn.commit()

                new_links = extract_internal_links(url, result, allowed_prefixes)
                for link in new_links:
                    if link not in seen_in_memory:
                        seen_in_memory.add(link)
                        queue.append((link, depth + 1))

                print(f"✅ Processed: {url} | Depth: {depth} | Queue: {len(queue)}")

        finish_ingestion_run(cur, run_id, "success", processed_count)
        conn.commit()

        print("\n🎉 Crawl complete")
        print(f"📄 Pages processed: {processed_count}")

    except Exception as e:
        print(f"💥 Fatal error: {e}")
        finish_ingestion_run(cur, run_id, "failed", processed_count)
        conn.commit()
        raise

    finally:
        cur.close()
        conn.close()

    return run_id

