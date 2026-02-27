# ingestion/utils/llm.py

import json
from openai import OpenAI

client = OpenAI()

# =========================================================
# 1️⃣ TOPIC SEGMENTATION (USED BY chunk_docs.py)
# =========================================================

SEGMENTATION_PROMPT = """
You are a senior technical documentation segmentation engine.

Your task:
Split the provided MARKDOWN DOCUMENTATION into CLEAR, LOGICAL TOPIC SECTIONS.

----------------------------------
OUTPUT FORMAT (STRICT):
Return a JSON ARRAY. Each item must be an object with:

- topic: short, human-readable title (string)
- content: exact markdown belonging to this topic (string)
- chunk_type: "article" or "hub"

----------------------------------
RULES:
- Use markdown headers as boundaries
- Do NOT summarize
- Do NOT invent content
- Remove navigation / footer noise
- Preserve markdown formatting
- Prefer fewer, meaningful chunks

Return ONLY valid JSON.
"""

def topic_chunk_markdown(markdown: str) -> list:
    if not markdown or len(markdown.strip()) < 500:
        return []

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SEGMENTATION_PROMPT},
            {"role": "user", "content": markdown}
        ],
        max_tokens=4096
    )

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue

        content = item.get("content", "").strip()
        if len(content) < 200:
            continue

        cleaned.append({
            "topic": item.get("topic", "").strip(),
            "content": content,
            "chunk_type": item.get("chunk_type", "article")
        })

    return cleaned


# =========================================================
# 2️⃣ CHUNK CLASSIFICATION (USED BY enrich_chunks.py)
# =========================================================

CLASSIFICATION_PROMPT = """
You are a strict documentation classification engine.

Given a documentation chunk, return a JSON object with:

- persona: list of applicable personas (choose ONLY from):
  ["NoCode", "LowCode", "ProDeveloper", "Admin", "Architect"]

- intent: one of:
  ["what-is", "how-to", "reference", "troubleshooting", "comparison", "why"]

- complexity: one of:
  ["no-code", "low-code", "pro-code"]

- confidence: float between 0 and 1

PERSONA DEFINITIONS:

NoCode:
Business users configuring features using UI only. No scripting.

LowCode:
Power users using expressions, formulas, connectors, configuration logic.

ProDeveloper:
Developers writing code, SDK usage, APIs, scripting, advanced integration.

Admin:
Tenant configuration, governance, policies, licensing, environment setup.

Architect:
System design, scalability, integration patterns, solution architecture.

Rules:
- Base decision ONLY on content
- Multiple personas allowed if clearly applicable
- Be conservative
- Return ONLY valid JSON
"""

def classify_chunk(text: str) -> dict:
    if not text or len(text.strip()) < 200:
        return {}

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": text}
        ]
    )

    raw = response.choices[0].message.content.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
