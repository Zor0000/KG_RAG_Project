# ingestion/enrich_chunks.py

import time
import re
from typing import Dict, List
from ingestion.db import get_connection

from ingestion.utils.llm import classify_chunk
from openai import OpenAI


# ============================================================
# 🔹 CONFIG
# ============================================================

ENRICHMENT_VERSION = 1

MODEL_NAME = "gpt-5.2"
PROMPT_VERSION = 1
TEMPERATURE = 0.0

LLM_SLEEP = 0.2
MIN_TEXT_LENGTH = 250

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
# 🔹 HELPERS
# ============================================================

def canonicalize_title(title: str) -> str:
    if not title:
        return ""
    title = title.lower()
    title = re.sub(r"(overview|introduction|getting started|guide)", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title.title()


def classify_super_topic(text: str) -> str:
    prompt = f"""
Select the SINGLE most appropriate super topic from the list.
Return ONLY the exact topic name.

Available topics:
{chr(10).join(SUPER_TOPICS)}

Text:
{text[:3000]}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a strict classification engine."},
            {"role": "user", "content": prompt},
        ],
    )

    topic = response.choices[0].message.content.strip()

    if topic not in SUPER_TOPICS:
        return "Agent Fundamentals"

    return topic


def normalize(meta: Dict, title: str, super_topic: str) -> Dict:
    """
    Strict normalization while preserving NoCode and LowCode.
    """

    persona = meta.get("persona", [])

    if isinstance(persona, str):
        persona = [persona]

    # Keep only allowed personas
    persona = [p for p in persona if p in ALLOWED_PERSONAS]

    # If LLM didn't return persona, default safely
    if not persona:
        persona = ["ProDeveloper"]

    intent = meta.get("intent")
    if intent not in ALLOWED_INTENTS:
        intent = "reference"

    complexity = meta.get("complexity", "pro-code")

    confidence = meta.get("confidence", 0.6)
    try:
        confidence = float(confidence)
    except:
        confidence = 0.6

    return {
        "persona": persona,
        "intent": intent,
        "complexity": complexity,
        "confidence": confidence,
        "canonical_topic": canonicalize_title(title),
        "super_topic": super_topic,
    }


# ============================================================
# 🔹 FETCH CHUNKS
# ============================================================

def fetch_chunks_to_enrich(source_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT c.id, c.chunk_text, c.topic
        FROM ingestion.chunks c
        JOIN ingestion.documents d ON d.id = c.document_id
        WHERE d.source_id = %s
        AND NOT EXISTS (
            SELECT 1
            FROM ingestion.enriched_chunks ec
            WHERE ec.chunk_id = c.id
            AND ec.enrichment_version = %s
        )
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
    cur = conn.cursor()

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
    print(f"🧠 Enriching {len(rows)} chunks for source: {source_id}\n")

    total_processed = 0
    batch = []

    try:
        for chunk_id, text, title in rows:

            if len(text) < MIN_TEXT_LENGTH:
                super_topic = classify_super_topic(text)

                data = {
                    "persona": ["ProDeveloper"],
                    "intent": "reference",
                    "complexity": "pro-code",
                    "confidence": 0.4,
                    "canonical_topic": canonicalize_title(title),
                    "super_topic": super_topic,
                }
            else:
                meta = classify_chunk(text)
                super_topic = classify_super_topic(text)
                data = normalize(meta or {}, title, super_topic)

            batch.append((
                chunk_id,
                data["persona"][0],  # store primary persona
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
                run_id
            ))

            if len(batch) >= 20:
                insert_enrichment_batch(batch)
                batch.clear()

            total_processed += 1
            time.sleep(LLM_SLEEP)

        if batch:
            insert_enrichment_batch(batch)

        print("\n🎉 Enrichment complete")
        print(f"🧩 Chunks enriched: {total_processed}")

        return total_processed

    except Exception:
        print(f"\n💥 Enrichment failed after {total_processed} chunks")
        raise