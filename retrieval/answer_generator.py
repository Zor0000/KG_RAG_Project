from openai import OpenAI
from typing import List, Dict
from collections import defaultdict

client = OpenAI()


# ============================================================
# 🔹 Final Answer Generator (Unchanged Behavior)
# ============================================================

def generate_answer(query: str,
                    chunks: List[Dict]) -> Dict:

    context_blocks = []
    sources = set()

    for i, c in enumerate(chunks, 1):
        context_blocks.append(f"[Source {i}]\n{c['text']}")
        if c.get("source_url"):
            sources.add(c["source_url"])

    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = """
You are a senior AI systems documentation expert.

Your job:
- Produce a technically strong, structured answer.
- Use ONLY the provided context.
- Never hallucinate.
- Write clearly and professionally.
- Use sections and bullet points where useful.
"""

    user_prompt = f"""
User Question:
{query}

Documentation Context:
{context}

Produce a detailed, structured, technically accurate answer.
"""

    response = client.chat.completions.create(
        model="gpt-5.2",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sorted(sources)
    }


# ============================================================
# 🔹 Guided Clarification Preview Generator
# ============================================================

def generate_guided_preview(query: str,
                            preview_chunks: List[Dict],
                            product_options: List[str]) -> Dict:

    """
    Generates:
    1️⃣ A short high-level explanation
    2️⃣ Brief breakdown per detected product
    3️⃣ Ends with clarification question
    """

    # Group chunks by product
    product_groups = defaultdict(list)
    sources = set()

    for c in preview_chunks:
        product = c.get("product", "unknown")
        product_groups[product].append(c["text"])
        if c.get("source_url"):
            sources.add(c["source_url"])

    # Build compact context grouped by product
    context_sections = []

    for product, texts in product_groups.items():
        combined = "\n\n".join(texts[:2])  # limit per product
        context_sections.append(f"[Product: {product}]\n{combined}")

    context = "\n\n---\n\n".join(context_sections)

    system_prompt = """
You are a senior AI documentation expert.

The user asked a question that applies to multiple products.

Your task:
1. Provide a short high-level explanation answering the question generically.
2. Briefly explain how the process differs across each product.
3. Do NOT hallucinate — use only provided context.
4. End by asking the user which product they want detailed steps for.
Keep the explanation concise but helpful.
"""

    user_prompt = f"""
User Question:
{query}

Context (Grouped by Product):
{context}

Available Products:
{", ".join(product_options)}

Generate:
- A helpful overview
- A short per-product explanation
- A final clarification question
"""

    response = client.chat.completions.create(
        model="gpt-5.2",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sorted(sources)
    }