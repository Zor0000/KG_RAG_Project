"""
query_rewriter.py
─────────────────
Rewrites the user's raw input into a clean, self-contained search query
before it hits unified_search().

Why this matters:
  - Users ask vague follow-ups: "how about for Power Automate?" — the
    rewriter resolves the reference using conversation history.
  - Users ask multi-part questions — the rewriter extracts the core intent.
  - Casual phrasing ("so like, what's the deal with...") gets cleaned up
    into a precise retrieval query the embedding model handles well.

The rewriter uses the conversation summary (long context) and the last
2 raw turns (short context) so references like "that", "it", "the one
you mentioned" are resolved correctly.
"""

import os
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def rewrite_query(
    raw_question:   str,
    summary:        str,
    recent_turns:   list[dict],
) -> str:
    """
    Returns a rewritten, self-contained search query.
    Falls back to the original question if the LLM call fails.

    Parameters
    ----------
    raw_question  : Exactly what the user typed.
    summary       : Redis session summary (may be empty on first turn).
    recent_turns  : Last N turns as dicts with 'question' and 'answer' keys.
    """

    # Build a compact history block from the last 2 turns only
    # (more than 2 adds noise, the summary covers the rest)
    history_lines = []
    for t in recent_turns[-2:]:
        history_lines.append(f"User: {t['question']}")
        history_lines.append(f"Agent: {t['answer'][:250]}...")
    history_block = "\n".join(history_lines)

    prompt = f"""
You are a search query optimizer for a Microsoft 365, Copilot Studio,
Power Platform, and Azure AI knowledge base.

Your job: rewrite the user's raw message into a single, precise, self-contained
search query that will retrieve the most relevant documents.

Rules:
- Resolve ALL pronouns and references ("it", "that", "the one you mentioned",
  "same as above") using the conversation history below.
- Remove filler words, greetings, and emotional language.
- Keep domain-specific terms exactly as-is (e.g. "Power Automate", "Copilot Studio",
  "Azure OpenAI", "RAG pipeline").
- If the question has multiple parts, focus on the PRIMARY information need.
- Output ONLY the rewritten query — no explanation, no prefix, no quotes.
- If the question is already clear and self-contained, return it unchanged.

--- Conversation summary ---
{summary or "(no previous context)"}

--- Recent exchanges ---
{history_block or "(first message)"}

--- User's raw message ---
{raw_question}

Rewritten query:"""

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120,
        )
        rewritten = resp.choices[0].message.content.strip()
        # Safety: if LLM returns empty or something very short, use original
        return rewritten if len(rewritten) > 5 else raw_question
    except Exception as e:
        print(f"Query rewrite error: {e} — using original query")
        return raw_question