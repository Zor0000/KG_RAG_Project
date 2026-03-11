# ingestion/enrich_chunks.py

import time
import re
import random
from typing import Dict, List
from ingestion.db import get_connection

from ingestion.utils.llm import classify_chunk
from openai import OpenAI


# ============================================================
# 🔹 CONFIG
# ============================================================

ENRICHMENT_VERSION = 1

MODEL_NAME     = "gpt-4o-mini"   # ← was "gpt-5.2" which does not exist
PROMPT_VERSION = 1
TEMPERATURE    = 0.0

LLM_SLEEP       = 0.3    # seconds between calls (avoid rate limits)
MIN_TEXT_LENGTH = 250

# Retry config for API errors (rate limits, timeouts, etc.)
MAX_RETRIES     = 5
RETRY_BASE_WAIT = 10     # seconds — doubles each retry (exponential backoff)

client = OpenAI()

SUPER_TOPICS = [
    "Agent Fundamentals",
    "Agent Authoring",
    "Agent Flows",
    "Generative Orchestration",
    "Prompt Engineering",
    "Bot SDK & Runtime",
    "Bot Channels & Direct Line",
    "Bot Hosting & Infrastructure",
    "Bot Security",
    "Agent Publishing & Deployment",
    "Agent Testing & Debugging",
    "Agent Lifecycle Management",
    "Connectors",
    "External System Integration",
    "Microsoft Teams Integration",
    "Microsoft 365 Integration",
    "Authentication & Authorization",
    "App Registration & Service Principals",
    "Workflow Governance",
    "Environment & Solutions",
    "Admin Center Management",
    "Licensing & Quotas",
    "Data Privacy & Compliance",
    "Knowledge Sources",
    "File & Structured Data Processing",
    "Telemetry & Logging",
    "Analytics & Reporting",
    "Performance Optimization",
    "Usage & Capacity Management",
    "Responsible AI",
    "Multi-Agent Systems",
    "Agent-to-Agent Communication",
    "Tool Execution & Function Calling",
    "Autonomous Agents",
    "Human-in-the-Loop Systems",
    "LLM Orchestration Patterns",
    "Code Execution & Sandboxing",
]

ALLOWED_PERSONAS = {"NoCode", "LowCode", "ProDeveloper", "Admin", "Architect"}

ALLOWED_INTENTS = {
    "what-is",
    "how-to",
    "why",
    "reference",
    "troubleshooting",
    "comparison",
}


# ============================================================
# 🔹 RETRY WRAPPER
# ============================================================

def call_with_retry(fn, *args, **kwargs):
    """
    Call any function that hits the OpenAI API.
    Retries up to MAX_RETRIES times with exponential backoff.
    Never crashes the whole job — returns None on total failure
    so the chunk gets a safe default and processing continues.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = "rate" in err or "429" in err or "quota" in err
            is_timeout    = "timeout" in err or "timed out" in err
            is_server_err = "500" in err or "503" in err or "server" in err

            if attempt == MAX_RETRIES:
                print(f"  ⚠️  Giving up after {MAX_RETRIES} attempts: {e}")
                return None

            if is_rate_limit or is_timeout or is_server_err:
                wait = RETRY_BASE_WAIT * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"  🔄 Attempt {attempt}/{MAX_RETRIES} failed ({type(e).__name__}). "
                      f"Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                # Non-retryable error (bad prompt, invalid model, etc.)
                print(f"  ❌ Non-retryable error: {e}")
                return None

    return None


# ============================================================
# 🔹 HELPERS
# ============================================================

def canonicalize_title(title: str) -> str:
    if not title:
        return ""
    title = title.lower()
    title = re.sub(r"(overview|introduction|getting started|guide)", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title.title()


def classify_all(text: str, title: str) -> dict:
    """
    ONE LLM call returns all enrichment fields at once.
    Replaces separate classify_chunk + classify_super_topic calls.
    2x faster — cuts API calls per chunk from 2 to 1.
    """
    super_topics_str = "\n".join(SUPER_TOPICS)

    def _call():
        import json
        prompt = f"""You are a strict classification engine for Microsoft Copilot documentation.

Analyse the text and return a JSON object with EXACTLY these fields:
{{
  "persona": one of ["NoCode","LowCode","ProDeveloper","Admin","Architect"],
  "intent": one of ["what-is","how-to","why","reference","troubleshooting","comparison"],
  "complexity": one of ["no-code","low-code","pro-code"],
  "confidence": float 0.0-1.0,
  "canonical_topic": clean 2-5 word topic title from "{title}" (no: overview/introduction/getting started/guide),
  "super_topic": EXACTLY one name from the list below
}}

Super topics:
{super_topics_str}

Return ONLY valid JSON. No markdown, no backticks, no explanation.

Text:
{text[:3000]}"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user",   "content": prompt},
            ],
        )
        import json
        return json.loads(response.choices[0].message.content.strip())

    result = call_with_retry(_call)

    if not result or not isinstance(result, dict):
        return {
            "persona": "ProDeveloper", "intent": "reference",
            "complexity": "pro-code", "confidence": 0.5,
            "canonical_topic": canonicalize_title(title),
            "super_topic": "Agent Fundamentals",
        }

    persona = result.get("persona", "ProDeveloper")
    if persona not in ALLOWED_PERSONAS:
        persona = "ProDeveloper"

    intent = result.get("intent", "reference")
    if isinstance(intent, list):
        intent = intent[0] if intent else "reference"
    if intent not in ALLOWED_INTENTS:
        intent = "reference"

    super_topic = result.get("super_topic", "Agent Fundamentals")
    if super_topic not in SUPER_TOPICS:
        super_topic = "Agent Fundamentals"

    confidence = result.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.5

    return {
        "persona":         persona,
        "intent":          intent,
        "complexity":      result.get("complexity", "pro-code"),
        "confidence":      confidence,
        "canonical_topic": result.get("canonical_topic") or canonicalize_title(title),
        "super_topic":     super_topic,
    }


def normalize(meta: Dict, title: str, super_topic: str) -> Dict:
    persona = meta.get("persona", [])
    if isinstance(persona, str):
        persona = [persona]
    persona = [p for p in persona if p in ALLOWED_PERSONAS]
    if not persona:
        persona = ["ProDeveloper"]

    intent = meta.get("intent")
    # classify_chunk sometimes returns intent as a list — flatten it
    if isinstance(intent, list):
        intent = intent[0] if intent else "reference"
    if not isinstance(intent, str) or intent not in ALLOWED_INTENTS:
        intent = "reference"

    complexity = meta.get("complexity", "pro-code")

    confidence = meta.get("confidence", 0.6)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.6

    return {
        "persona":         persona,
        "intent":          intent,
        "complexity":      complexity,
        "confidence":      confidence,
        "canonical_topic": canonicalize_title(title),
        "super_topic":     super_topic,
    }


# ============================================================
# 🔹 FETCH CHUNKS
# ============================================================

def fetch_chunks_to_enrich(source_id):
    """
    Only returns chunks NOT already in enriched_chunks.
    Re-running after a crash automatically resumes from where it stopped.
    """
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
        SELECT c.id, c.chunk_text, c.topic
        FROM ingestion.chunks c
        JOIN ingestion.documents d ON d.id = c.document_id
        WHERE d.source_id = %s
        AND NOT EXISTS (
            SELECT 1 FROM ingestion.enriched_chunks ec
            WHERE ec.chunk_id = c.id
            AND ec.enrichment_version = %s
        )
        ORDER BY c.id   -- consistent order so resume is deterministic
    """, (source_id, ENRICHMENT_VERSION))

    rows = cur.fetchall()
    conn.close()
    return rows


# ============================================================
# 🔹 BULK INSERT
# ============================================================

def insert_enrichment_batch(records: List[tuple]):
    if not records:
        return

    conn = get_connection()
    cur  = conn.cursor()

    cur.executemany("""
        INSERT INTO ingestion.enriched_chunks (
            id,
            chunk_id,
            persona,
            intent,
            complexity,
            confidence,
            canonical_topic,
            super_topic,
            enrichment_version,
            model_name,
            prompt_version,
            temperature,
            drift_flag,
            ingestion_run_id,
            created_at
        )
        VALUES (
            gen_random_uuid(),
            %s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,
            NOW()
        )
        ON CONFLICT DO NOTHING
    """, records)

    conn.commit()
    conn.close()


# ============================================================
# 🔹 MAIN
# ============================================================

def main(source_id, run_id=None):

    rows = fetch_chunks_to_enrich(source_id)
    total_remaining = len(rows)
    conn_c = get_connection()
    cur_c  = conn_c.cursor()
    cur_c.execute(
        "SELECT COUNT(*) FROM ingestion.enriched_chunks ec "
        "JOIN ingestion.chunks c ON c.id = ec.chunk_id "
        "JOIN ingestion.documents d ON d.id = c.document_id "
        "WHERE d.source_id = %s AND ec.enrichment_version = %s",
        (source_id, ENRICHMENT_VERSION)
    )
    already_done = cur_c.fetchone()[0]
    conn_c.close()
    total_all = already_done + total_remaining
    print(f"🧠 Enrichment status for: {source_id}")
    print(f"   ✅ Already done : {already_done}/{total_all}")
    print(f"   ⏳ Remaining    : {total_remaining}/{total_all}")
    print(f"   ⚡ Est. time    : ~{int(total_remaining * 0.6 / 60)} mins\n")
    if total_remaining == 0:
        print("✅ All chunks already enriched — nothing to do.")
        return 0

    total_processed = 0
    batch           = []

    try:
        for chunk_id, text, title in rows:

            if len(text) < MIN_TEXT_LENGTH:
                data = {
                    "persona": "ProDeveloper", "intent": "reference",
                    "complexity": "pro-code", "confidence": 0.4,
                    "canonical_topic": canonicalize_title(title),
                    "super_topic": "Agent Fundamentals",
                }
            else:
                # Single LLM call — 2x faster than before
                data = classify_all(text, title)

            persona_val = data["persona"]
            if isinstance(persona_val, list):
                persona_val = persona_val[0] if persona_val else "ProDeveloper"

            batch.append((
                chunk_id,
                persona_val,
                data["intent"],
                data["complexity"],
                data["confidence"],
                data["canonical_topic"],
                data["super_topic"],
                ENRICHMENT_VERSION,
                MODEL_NAME,
                PROMPT_VERSION,
                TEMPERATURE,
                False,
                run_id,
            ))

            # Flush every 20 chunks so a crash never loses more than 20
            if len(batch) >= 20:
                insert_enrichment_batch(batch)
                batch.clear()

            total_processed += 1

            # Progress log every 100 chunks
            if total_processed % 100 == 0:
                pct = (total_processed / total_remaining) * 100
                print(f"  📊 {total_processed}/{total_remaining} remaining ({pct:.1f}%) — total done: {already_done + total_processed}/{total_all}")

            time.sleep(LLM_SLEEP)

        # Flush remaining
        if batch:
            insert_enrichment_batch(batch)

        print("\n🎉 Enrichment complete")
        print(f"🧩 Chunks enriched: {total_processed}")
        return total_processed

    except Exception as e:
        # Flush whatever is in the batch before raising
        # so we don't lose the last <20 chunks on crash
        if batch:
            try:
                insert_enrichment_batch(batch)
                print(f"  💾 Saved {len(batch)} chunks before crash")
            except Exception:
                pass
        print(f"\n💥 Enrichment failed after {total_processed} chunks")
        raise