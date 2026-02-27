import re
import hashlib
from ingestion.db import get_connection
from tqdm import tqdm
import uuid
def clean_topic(title: str) -> str:
    if not title:
        return "Unknown"

    # Remove markdown links [text](url)
    title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)

    # Remove inline markdown artifacts
    title = re.sub(r"[#*_`]", "", title)

    # Remove excessive whitespace
    title = re.sub(r"\s+", " ", title).strip()

    # Cap length to prevent Milvus overflow
    return title[:300]
# ============================================================
# 🔹 CONFIG
# ============================================================

CHUNKING_VERSION = 1

MIN_CHUNK_LEN = 300
MAX_CHUNK_LEN = 1200


# ============================================================
# 🔹 HELPERS
# ============================================================

def clean_body_text(body: str) -> str:
    body = re.sub(r"!\[.*?\]\(.*?\)", "", body)
    body = re.sub(r"\[\]\(.*?\)", "", body)
    return body.strip()


def is_low_signal_section(title: str, body: str) -> bool:
    title_lower = title.lower()

    JUNK_TITLES = [
        "preview",
        "feedback",
        "in this article",
        "table of contents",
        "share via"
    ]

    if any(j in title_lower for j in JUNK_TITLES):
        return True

    text_without_links = re.sub(r"\[.*?\]\(.*?\)", "", body)

    if len(text_without_links.strip()) < 200:
        return True

    link_count = len(re.findall(r"\[.*?\]\(.*?\)", body))
    line_count = len(body.splitlines())

    if line_count > 0 and (link_count / line_count) > 0.6:
        return True

    return False


def split_by_sections(markdown: str):
    parts = re.split(r"\n##\s+", markdown)
    results = []

    for part in parts:
        lines = part.strip().splitlines()
        if not lines:
            continue

        title = lines[0].strip("# ").strip()
        body = "\n".join(lines[1:]).strip()

        if len(body) < MIN_CHUNK_LEN:
            continue

        if is_low_signal_section(title, body):
            continue

        body = clean_body_text(body)
        cleaned_title = clean_topic(title)
        results.append((cleaned_title, body[:MAX_CHUNK_LEN]))

    return results


# ============================================================
# 🔹 FETCH DOCUMENTS
# ============================================================

def fetch_documents_to_chunk(cur, source_id):

    cur.execute("""
        SELECT d.id, cd.cleaned_text
        FROM ingestion.cleaned_documents cd
        JOIN ingestion.documents d
          ON d.id = cd.document_id
        WHERE d.source_id = %s
    """, (source_id,))

    return cur.fetchall()


# ============================================================
# 🔹 MAIN
# ============================================================

def main(source_id, run_id=None):

    print(f"🚀 Starting chunking for source: {source_id}")

    conn = get_connection()
    cur = conn.cursor()

    rows = fetch_documents_to_chunk(cur, source_id)

    total_chunks = 0
    processed_docs = 0

    print(f"📄 Found {len(rows)} cleaned documents\n")

    for document_id, markdown in tqdm(rows, desc="📄 Chunking"):

        if not markdown or len(markdown) < 500:
            continue

        try:
            # Always delete existing chunks (safe for rebuild per source)
            cur.execute("""
                DELETE FROM ingestion.chunks
                WHERE document_id = %s
            """, (document_id,))

            doc_title = markdown.splitlines()[0].lstrip("# ").strip()
            topic = clean_topic(doc_title)

            sections = split_by_sections(markdown)

            for idx, (section_title, content) in enumerate(sections):

                chunk_hash = hashlib.sha256(
                    f"{document_id}|{section_title}|{content}".encode("utf-8")
                ).hexdigest()

                chunk_id = str(uuid.uuid4())

                cur.execute("""
                    INSERT INTO ingestion.chunks (
                        id,
                        document_id,
                        chunk_index,
                        section_title,
                        topic,
                        chunk_text,
                        chunk_hash,
                        chunk_type,
                        chunking_version,
                        ingestion_run_id,
                        created_at
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        'section',
                        %s,
                        %s,
                        NOW()
                    )
                    ON CONFLICT DO NOTHING
                """, (
                    chunk_id,
                    document_id,
                    idx,
                    section_title,
                    topic,
                    content,
                    chunk_hash,
                    CHUNKING_VERSION,
                    run_id
                ))

                total_chunks += 1

            processed_docs += 1

            # Commit every 50 documents
            if processed_docs % 50 == 0:
                conn.commit()

        except Exception as e:
            conn.rollback()
            print(f"❌ Failed chunking document {document_id}: {e}")

    conn.commit()
    cur.close()
    conn.close()

    print("\n🎉 Chunking complete")
    print(f"🧩 Total chunks created: {total_chunks}")