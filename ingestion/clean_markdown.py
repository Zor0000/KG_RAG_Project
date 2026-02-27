import hashlib
import re
from ingestion.db import get_connection


# ============================================================
# 🔹 CONFIG
# ============================================================

CLEANING_VERSION = 1

MIN_GOOD_LENGTH = 200
MIN_LOW_SIGNAL_LENGTH = 80


# ============================================================
# 🔹 CLEANING LOGIC
# ============================================================

def extract_article_markdown(text: str) -> str:
    if not text or len(text.strip()) < 100:
        return ""

    text = text.replace("\r\n", "\n")

    title_match = re.search(r"(^|\n)#\s+.+", text)
    if not title_match:
        return ""

    text = text[title_match.start():]

    text = re.sub(r"Summarize this article for me", "", text, flags=re.I)
    text = re.sub(r"Ask Learn.*", "", text, flags=re.I)
    text = re.sub(r"Access to this page requires authorization.*", "", text, flags=re.I)

    text = re.sub(r"##\s+In this article[\s\S]+?(?=\n##|\Z)", "", text, flags=re.I)
    text = re.sub(r"##\s+Ask Learn[\s\S]+?(?=\n##|\Z)", "", text, flags=re.I)
    text = re.sub(r"##\s+Additional resources[\s\S]+?(?=\n##|\Z)", "", text, flags=re.I)

    STOP_MARKERS = [
        "\n## Feedback",
        "\nWas this page helpful",
        "\nTheme\n",
        "\nYour Privacy Choices",
        "\nAI Disclaimer",
        "\nPrevious Versions",
        "\nBlog",
        "\nContribute",
        "\nPrivacy",
        "\nTerms of Use",
        "\n© Microsoft",
    ]

    for marker in STOP_MARKERS:
        if marker in text:
            text = text.split(marker)[0]

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


# ============================================================
# 🔹 FETCH DOCUMENTS TO CLEAN
# ============================================================

def fetch_documents_to_clean(cur, source_id):

    cur.execute("""
        SELECT d.id, rp.raw_content, d.version
        FROM ingestion.documents d
        JOIN ingestion.raw_pages rp
            ON d.id = rp.document_id
        WHERE d.source_id = %s
    """, (source_id,))

    return cur.fetchall()


# ============================================================
# 🔹 MAIN
# ============================================================

def main(source_id, run_id=None):

    print(f"🧹 Starting markdown cleaning for source: {source_id}")

    conn = get_connection()
    cur = conn.cursor()

    rows = fetch_documents_to_clean(cur, source_id)

    good, low, empty = 0, 0, 0
    processed = 0

    print(f"🧾 Found {len(rows)} documents\n")

    for document_id, raw_content, doc_version in rows:

        cleaned = extract_article_markdown(raw_content)

        if not cleaned:
            empty += 1
            continue

        length = len(cleaned)

        if length < MIN_LOW_SIGNAL_LENGTH:
            empty += 1
            continue

        if length < MIN_GOOD_LENGTH:
            low += 1
        else:
            good += 1

        markdown_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

        try:
            # Delete old cleaned version
            cur.execute("""
                DELETE FROM ingestion.cleaned_documents
                WHERE document_id = %s
            """, (document_id,))

            # Insert new cleaned version
            cur.execute("""
                INSERT INTO ingestion.cleaned_documents (
                    id,
                    document_id,
                    cleaned_text,
                    markdown_hash,
                    cleaning_version,
                    ingestion_run_id,
                    created_at
                )
                VALUES (
                    gen_random_uuid(),
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    NOW()
                )
            """, (
                document_id,
                cleaned,
                markdown_hash,
                CLEANING_VERSION,
                run_id
            ))

            processed += 1

            # Commit every 100 docs
            if processed % 100 == 0:
                conn.commit()

        except Exception as e:
            conn.rollback()
            print(f"❌ Failed cleaning {document_id}: {e}")

    conn.commit()
    cur.close()
    conn.close()

    print("\n🎉 Markdown cleaning complete")
    print(f"✅ Good articles: {good}")
    print(f"⚠️ Low-signal articles (kept): {low}")
    print(f"❌ Empty discarded: {empty}")